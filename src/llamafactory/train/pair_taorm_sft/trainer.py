# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import Trainer
from typing_extensions import override
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import math
from contextlib import nullcontext
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
import time

from trl.models import PreTrainedModelWrapper
from trl.trainer import disable_dropout_in_model
from ...extras.logging import get_logger
from ..callbacks import FixValueHeadModelCallback, PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_outs_info, get_scores, get_scores2
from ...extras.constants import IGNORE_INDEX
from accelerate.utils import is_deepspeed_available
if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class DummyScheduler(torch.optim.lr_scheduler.LRScheduler):
    
    def __init__(self, base_schedule: torch.optim.lr_scheduler.LRScheduler):
        self.base_schedule = base_schedule

    @override
    def step(self, epoch: Optional[int] = None):
        results = self.base_schedule.step(epoch)
        for param_group in self.base_schedule.optimizer.param_groups:
            if param_group["group_num"] == 3:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(1e-5)
                else:
                    param_group["lr"] = 1e-5
        return results

    def state_dict(self):
        return self.base_schedule.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        return self.base_schedule.load_state_dict(state_dict)
    
    def get_last_lr(self) -> List[float]:
        return self.base_schedule.get_last_lr()
    
    def get_lr(self) -> List[float]:
        return self.base_schedule.get_lr()
    
    def print_lr(
        self,
        is_verbose: bool,
        group: Dict[str, Any],
        lr: float,
        epoch: Optional[int] = None,
    ):
        return self.base_schedule.print_lr(is_verbose, group, lr, epoch)

    
class MaskGoalTrainer(Trainer):
    r"""
    Inherits Trainer to compute masked-goal loss.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", ref_model, processor: Optional["ProcessorMixin"], stage:str, **kwargs #ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]], 
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.training_args = kwargs["args"]
        self.can_return_loss = True  # override property to return eval_loss
        self.stage = stage
        
        self.ref_model = ref_model
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()
        
        self.add_callback(FixValueHeadModelCallback)
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        
        json_file_path = './data/special tokens.json'
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        _POS = "Yes"#data["output"]["postive"]
        _NEG = "No"#data["output"]["negtive"]
        _NAT = data["output"]["natural"]

        self.pos_token = self.tokenizer.encode(f"{_POS}")[-1]
        self.neg_token = self.tokenizer.encode(f"{_NEG}")[-1]
        self.nat_token = self.tokenizer.encode(f"{_NAT}")[-1]

        self.use_pos_only=False
        self.beta = 0.1
        self.weight_ratio = finetuning_args.weight_ratio #0.01
        self.reg_ratio = finetuning_args.reg_ratio #10
        self.radis = finetuning_args.radis
        self.use_pref = finetuning_args.use_pref
        self.use_log = finetuning_args.use_log
        self.use_align = finetuning_args.use_align
        self.no_logsigmoid_sft = True
        self.step = 0
        self.smooth_weight = 0.5
        self.smooth_weight1 = 0.5
        self.smooth_weight2 = 0.5
        self.state_path = os.path.join(self.training_args.output_dir, "states.txt")
        self.weight1 = 1.0
        self.weight11 = 1.0
        self.weight22 = 1.0
        self.weight2 = 1.0
        self.reward1 = 0
        self.reward2 = 0
    
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        #super().create_optimizer()
        self.create_ours_optimizer()
        return self.create_ours_optimizer()
        #return super().create_optimizer()

    def create_ours_optimizer(self):
        from transformers.trainer import is_sagemaker_mp_enabled
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and ("first_head" not in n and "second_head" not in n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "group_num": 1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and ("first_head" not in n and "second_head" not in n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "group_num": 2,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (("first_head" in n or "second_head" in n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "group_num": 3,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None, return_outputs: bool = False
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return DummyScheduler(super().create_scheduler(num_training_steps, optimizer))


    def set_first_one_to_zero(self, tensor):
        result = tensor.clone()
        
        for i in range(tensor.size(0)):
            first_one_idx = (tensor[i] == 1).nonzero(as_tuple=True)[0]
            
            if first_one_idx.numel() > 0:
                result[i][first_one_idx[0]] = 0
        
        return result

    def keep_last_k_percent_batch(self, mask_batch, k, zero_thres):
        if k == 100:
            return mask_batch
        new_mask_batch = torch.zeros_like(mask_batch)
        
        for i in range(mask_batch.size(0)):
            mask = mask_batch[i]

            if mask.sum() < zero_thres:
                continue
            
            ones_indices = torch.nonzero(mask == 1).squeeze()
            
            num_ones = len(ones_indices)
            num_to_keep = int(num_ones * k / 100)
            
            if num_to_keep == 0:
                continue
            
            indices_to_keep = ones_indices[-num_to_keep:]
        
            new_mask_batch[i, indices_to_keep] = 1
        
        return new_mask_batch

    def set_first_one_per_row(self, tensor):
        result = torch.zeros_like(tensor)
        
        for i in range(tensor.size(0)):
            first_one_idx = (tensor[i] == 1).nonzero(as_tuple=True)[0]
            if first_one_idx.numel() > 0:
                result[i][first_one_idx[0]] = 1
        return result

    def shift_log(self, x, eps=1e-6):
        return torch.log(x + eps)
    
    def binary_cross_entropy(self, input, target, weight=None, reduction='mean'):
        if not (input.min() >= 0 and input.max() <= 1):
            raise ValueError("Input values must be in [0, 1] range. Did you forget sigmoid?")
        
        loss = - (target * self.shift_log(input) + (1 - target) * self.shift_log(1 - input))
        
        if weight is not None:
            loss = loss * weight
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid reduction mode. Use 'mean', 'sum', or 'none'.")

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        
        reward_mask = (inputs["labels"] != IGNORE_INDEX).float()
        for idx, mask in enumerate(reward_mask):
            indices = torch.where(mask==1)[0]
            last_index = indices[-1] 
            mask[:last_index] = 0
            reward_mask[idx] = mask
            assert sum(mask) == 1

        batch_size = inputs["input_ids"].size(0) // 2
        (all_base_logits, all_sed_logits), _, rewards = model(**inputs, return_dual_logits=True, return_dict=True, use_cache=False, pos_gradient_alpha=self.weight_ratio, neg_gradient_alpha=0, train_sft=2.)
        all_base_logits = all_base_logits.to(torch.float32)
        
        all_base_logits_info = get_batch_outs_info(logits=all_base_logits, labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)
        outs_info = all_base_logits_info
        #all_base_logps = all_base_logits_info["logps"]
        all_fst_logps = all_base_logits_info["logps"]
        valid_length = all_base_logits_info["valid_length"]
        chosen_fst_logps, reject_fst_logps = all_fst_logps.split(batch_size, dim=0)
        chosen_length, reject_length = valid_length.split(batch_size, dim=0)
        #chosen_length = all_base_logits_info["valid_length"]

        rewards = rewards.to(torch.float32)
        #all_logits = model(**inputs, return_dict=True, use_cache=False).logits.to(torch.float32)
        #all_fst_logits = all_fst_logits.to(torch.float32)
        all_sed_logits = all_sed_logits.to(torch.float32)
        #all_third_logits = all_third_logits.to(torch.float32)
        #outs_info = get_batch_outs_info(logits=all_fst_logits, labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)
        #all_fst_logps = outs_info["logps"]
        

        sed_outs_info = get_batch_outs_info(logits=all_sed_logits, labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)
        all_sed_logps = sed_outs_info["logps"]

        #all_third_outs_info = get_batch_outs_info(logits=all_third_logits, labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)
        #all_third_logps = all_third_outs_info["logps"]

        #chosen_fst_logps, reject_fst_logps = all_fst_logps.split(batch_size, dim=0)
        #chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        #fst_chosen_length, reject_length = valid_length.split(batch_size, dim=0)
        chosen_rewards, reject_rewards = rewards.split(batch_size, dim=0)
        _, sed_rejected_logps = all_sed_logps.split(batch_size, dim=0)

        
    
        kto_tag = torch.zeros(inputs["input_ids"].size(0)).to(rewards.device).to(torch.float32)
        kto_tag[:batch_size] = 1
        
        dpo_loss = -(chosen_fst_logps / chosen_length).mean() - (sed_rejected_logps / chosen_length).mean() #- (sed_rejected_logps / reject_length).mean()#
        loss = dpo_loss
        #else:
        #    loss = self.weight_ratio * (dpo_loss + 0 * align_loss) + (1 - self.weight_ratio) * reward_loss

        if return_outputs:
            loss = dpo_loss
            #return loss, [logits, torch.sigmoid((rewards[:,:-1] * reward_mask[:,1:]).sum(-1)) * torch.exp(logps/valid_length - sed_outs_info["logps"] / valid_length), kto_tag] # + 
            return loss, [all_base_logits, (rewards * reward_mask).sum(-1), kto_tag]
            #return loss, [logits, torch.sigmoid((rewards[:,:-1] * loss_mask).mean(-1)), kto_tag]
            #return loss, [logits, torch.sigmoid(-sed_outs_info["logps"] / sed_outs_info["valid_length"]), kto_tag]
            #return loss, [logits, torch.exp(logps/valid_length), kto_tag]

        return loss
    
    def compute_score(self, logps, mask):
        assert sum(sum(mask)) == len(mask)
        return (logps * mask).sum(-1)
    
    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_score, kto_tag = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, tag in zip(chosen_score, kto_tag):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "tag": round(float(tag), 2)}))

            writer.write("\n".join(res))
    
    def win_ratio(self, y_true, y_pred): 
        positive_scores = y_pred[y_true == 1]  
        negative_scores = y_pred[y_true == 0]  

        count_positive_greater_than_negative = sum(positive_scores[:, None] > negative_scores)

        total_positive = len(positive_scores)
        total_negative = len(negative_scores)

        if isinstance(count_positive_greater_than_negative, int):
            ratio = count_positive_greater_than_negative / (total_positive * total_negative + 1e-6)
        else:
            ratio = sum(count_positive_greater_than_negative) / (total_positive * total_negative + 1e-6)
        return ratio

    def compute_best_of_N_accuracy(self, predict_results: "PredictionOutput", N=64) -> None:
        r"""
        Compute best-of-N results
        """
        logger.info(f"Computing best-of-N results, where N is " + str(N))
        chosen_score, kto_tag = predict_results.predictions
        assert len(kto_tag) % N == 0
        accuracy_list = []
        baseline_acc = []
        max_acc = []
        win_ratio = []
        for group_num in range(len(kto_tag) // N):
            inner_score = chosen_score[group_num*N:group_num*N+N]
            best_idx = torch.tensor(inner_score).max(0)[1]
            accuracy =  kto_tag[group_num*N+best_idx] == True
            accuracy_list.append([best_idx, accuracy])
            baseline_acc.append(torch.tensor(kto_tag[group_num*N:group_num*N+N] == True).float().mean())
            max_acc.append(float(kto_tag[group_num*N:group_num*N+N].any()))
            win_ratio.append(self.win_ratio(kto_tag[group_num*N:group_num*N+N], chosen_score[group_num*N:group_num*N+N]))
        
        logger.info(f"The random accuracy of best-of-N results is " + str(sum(baseline_acc).item() / len(baseline_acc)))
        logger.info(f"The maximum accuracy of best-of-N results is " + str(sum(max_acc) / len(baseline_acc)))
        logger.info(f"The reward accuracy of best-of-N results is " + str(sum([ele[1] for ele in accuracy_list]) / len(accuracy_list)))
        logger.info(f"The average winning score is " + str(sum(win_ratio).item() / len(win_ratio)))
    
    def compute_accuracy(self, predict_results: "PredictionOutput", threshold = 0.5) -> None:
        r"""
        Compute accuracy
        """
        logger.info(f"Computing the accuracy")
        chosen_score, kto_tag = predict_results.predictions
        labels = chosen_score >= threshold
        accuarcy = (labels==kto_tag).mean()
        precision = ((labels==kto_tag) * (labels==True)).sum() / sum(labels == True)
        recall = ((labels==kto_tag) * (labels==True)).sum()  / sum(kto_tag == True)
        auc = roc_auc_score(kto_tag, chosen_score)
        logger.info(f"The pos/neg ratio of labels is " + str(sum(kto_tag==True) / sum(kto_tag==False)))
        logger.info(f"The pos/neg ratio of predictions is " + str(sum(labels==True) / sum(labels==False)))
        logger.info(f"The accuarcy is " + str(accuarcy))
        logger.info(f"The precision is " + str(precision))
        logger.info(f"The recall is " + str(recall))
        logger.info(f"The area under curve (AUC) is " + str(auc))
        
    
        

