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
import random
from contextlib import nullcontext
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
import ast
from torch.nn.utils.rnn import pad_sequence

from trl.models import PreTrainedModelWrapper
from trl.trainer import disable_dropout_in_model
from ...extras.logging import get_logger
from ..callbacks import FixValueHeadModelCallback, PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_outs_info, get_label_probabilities
from ...extras.constants import IGNORE_INDEX
from accelerate.utils import is_deepspeed_available
if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)



class MaskGoalTrainer(Trainer):
    r"""
    Inherits Trainer to compute masked-goal loss.
    """

    def __init__(
        self, ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]], finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], stage:str, **kwargs
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

        #if stage == "orm_dg":
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
        self._REQ = data["input"]["request"]

        self.use_pos_only=False
        self.beta = 0.1
        self.weight_ratio = finetuning_args.weight_ratio #0.01
        self.reg_ratio = finetuning_args.reg_ratio #10
        self.temp_ratio = finetuning_args.temp_ratio
        self.radis = finetuning_args.radis
        self.use_pref = finetuning_args.use_pref
        self.use_log = finetuning_args.use_log
        self.use_align = finetuning_args.use_align
        self.no_logsigmoid_sft = True
        self.step = 0
        self.smooth_weight = 0.5

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
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None, return_outputs: bool = False
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_distance(self, m1, fix_m2, random_reg=0.5):
        parmas = []
        for p1, p2 in zip(m1.parameters(), fix_m2.parameters()):
            if not p1.requires_grad:
                continue
            if random.random() > random_reg:
                continue
            parmas.append((p1 - p2.detach()).flatten())
        return torch.cat(parmas).norm(p=2)

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        return self.dg_compute_loss(model, inputs, return_outputs)



    def keep_last_k_percent_batch(self, mask_batch, k):
        if k == 100:
            return mask_batch
        new_mask_batch = torch.zeros_like(mask_batch)
        
        for i in range(mask_batch.size(0)):
            mask = mask_batch[i]
            
            ones_indices = torch.nonzero(mask == 1).squeeze()
            
            num_ones = len(ones_indices)
            num_to_keep = int(num_ones * k / 100)
            
            if num_to_keep == 0:
                continue
            
            indices_to_keep = ones_indices[-num_to_keep:]
        
            new_mask_batch[i, indices_to_keep] = 1
        
        return new_mask_batch
    
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


    def dg_compute_loss(self, model, inputs, return_outputs=False):
        self.step += 1

        kto_tag = inputs.pop('kto_tag', None)
        kto_tag = kto_tag.float()
        step_val = inputs.pop('outlier_tag', None)
        idx = inputs.pop('idx', None)

        #if self.step % (1000 * self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps)== 0:
        #    with open(self.state_path, 'a', encoding='utf-8') as f:
        #        f.write(str(time.localtime(time.time()))+"\n")
        #        f.write(str(self.step)+"\n")
        #        f.write(str(self.smooth_weight)+"\n")
        reward_mask = (inputs["labels"] != IGNORE_INDEX).float()
        for idx, mask in enumerate(reward_mask):
            indices = torch.where(mask==1)[0]
            last_index = indices[-1] 
            mask[:last_index] = 0
            reward_mask[idx] = mask
            assert sum(mask) == 1

        batch_size = inputs["input_ids"].size(0) // 2
        if self.reg_ratio != 0:
            with torch.no_grad():
                all_base_logits = self.ref_model(**inputs, return_dict=True, use_cache=False).logits
        (train_all_base_logits, _), _, rewards = model(**inputs, return_dual_logits=True, return_dict=True, use_cache=False, pos_gradient_alpha=self.weight_ratio, neg_gradient_alpha=0)
        
        
        if self.reg_ratio != 0:
            all_base_logits = all_base_logits.to(torch.float32)
            all_base_logits_info = get_batch_outs_info(logits=all_base_logits, labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)
            outs_info = all_base_logits_info
            #all_base_logps = all_base_logits_info["logps"]

            all_fst_logps = all_base_logits_info["logps"]
            valid_length = all_base_logits_info["valid_length"]
            #chosen_fst_logps, reject_fst_logps = all_fst_logps.split(batch_size, dim=0)
            #chosen_length, reject_length = valid_length.split(batch_size, dim=0)
        #chosen_length = all_base_logits_info["valid_length"]

        rewards = rewards.to(torch.float32)
        #all_logits = model(**inputs, return_dict=True, use_cache=False).logits.to(torch.float32)

        
        reward_loss = (nn.functional.binary_cross_entropy(torch.sigmoid((rewards * reward_mask).sum(-1)), kto_tag, reduction="none")).mean() # #

        align_loss = 0
        if self.reg_ratio != 0:
            overall_score =  2*get_label_probabilities(all_base_logits, labels=inputs["labels"], temperature=0.7, top_p=0.8) #get_scores(all_base_logits, all_sed_logits, labels=inputs["labels"], kto_tag=kto_tag)
            loss_mask = self.set_first_one_to_zero(outs_info["loss_mask"])
            retain_mask = self.keep_last_k_percent_batch(loss_mask, 100, 1)

            s_rewards = torch.sigmoid(rewards)
            score2 = - kto_tag.unsqueeze(1) * (self.shift_log(s_rewards[:, :-1]) * s_rewards[:, 1:].detach()
                                                    + s_rewards[:, :-1].detach() * self.shift_log(s_rewards[:, 1:])) / 2
            score2 += - (1 - kto_tag).unsqueeze(1) * (self.shift_log(1 - s_rewards[:, :-1]) * (1 - s_rewards[:, 1:].detach())
                                                    + (1 - s_rewards[:, :-1].detach()) * self.shift_log(1 - s_rewards[:, 1:])) / 2
                                
            weight = (((s_rewards[:,1:] + s_rewards[:,:-1]) * kto_tag.unsqueeze(1) + (1-s_rewards[:,1:] + 1-s_rewards[:,:-1]) * (1 - kto_tag).unsqueeze(1))/2 * overall_score.detach() * retain_mask).sum().detach() / (retain_mask.sum() + 1e-6)#  (torch.cat([, rev_difference_value[:,1:] +rev_difference_value[:,:-1] ])/2 * overall_score.detach() * retain_mask).sum().detach() / (retain_mask.sum() + 1e-6)
            #if self.radis == 0.5:
            #    weight = (overall_score.detach() * retain_mask).sum().detach() / (retain_mask.sum() + 1e-6)
            if not return_outputs:
                self.smooth_weight = self.smooth_weight * 0.95 + weight * 0.05
            align_loss = (score2 * overall_score.detach() * retain_mask).sum() / (retain_mask.sum() + 1e-6) / (self.smooth_weight + 1e-6)

 
        train_all_base_logits_info = get_batch_outs_info(logits=train_all_base_logits.to(torch.float32), labels=inputs["labels"], candidate=[self.pos_token, self.neg_token], return_candidate=False)

        dpo_loss = - (train_all_base_logits_info["logps"][:batch_size] / train_all_base_logits_info["valid_length"][:batch_size]).mean()
        loss = dpo_loss + self.reg_ratio * align_loss + (1-self.reg_ratio) * reward_loss
        #else:
        #    loss = self.weight_ratio * (dpo_loss + 0 * align_loss) + (1 - self.weight_ratio) * reward_loss

        if return_outputs:
            loss = reward_loss
            #return loss, [logits, torch.sigmoid((rewards[:,:-1] * reward_mask[:,1:]).sum(-1)) * torch.exp(logps/valid_length - sed_outs_info["logps"] / valid_length), kto_tag] # + 
            return loss, [rewards, (rewards * reward_mask).sum(-1), kto_tag]
            #return loss, [logits, torch.sigmoid((rewards[:,:-1] * loss_mask).mean(-1)), kto_tag]
            #return loss, [logits, torch.sigmoid(-sed_outs_info["logps"] / sed_outs_info["valid_length"]), kto_tag]
            #return loss, [logits, torch.exp(logps/valid_length), kto_tag]
        return loss

    @torch.no_grad()
    def ema_update(self, updated_model, ref_model, decay, cuda=True, pass_bufer=False):
        update_fn = lambda e, m: decay * e + (1. - decay) * m
        for ema_v, model_v in zip(updated_model.parameters(), ref_model.parameters()):
            if cuda:
                model_v = model_v.cuda()  # to(device=self.device)
            ema_v.copy_(update_fn(ema_v, model_v))
        if not pass_bufer:
            for ema_v, model_v in zip(updated_model.buffers(), ref_model.buffers()):
                if cuda:
                    model_v = model_v.cuda()  # to(device=self.device)
                ema_v.copy_(model_v)

    def compute_score(self, logps, mask):
        assert sum(sum(mask)) == len(mask)
        return (logps * mask).sum(-1)
    
    def find_outlier_with_features(self, features: torch.Tensor, ratio: float=0.5):
        lens = features.shape[1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=lens//10)
        X_pca = pca.fit_transform(X_scaled)

        iso_forest = IsolationForest(contamination=ratio)  
        y_pred = iso_forest.fit_predict(X_pca)

        return torch.tensor(y_pred == -1).float()
    
    def find_outlier_with_scores(self, scores: torch.Tensor, ratio: float=0.5):
        k_elements = int(len(scores) * ratio)
        _, indices = torch.topk(scores, k_elements, largest=False)
        mask = torch.zeros_like(scores, dtype=torch.float)
        mask[indices] = 1
        return mask

    def save_pred_features(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "outlier_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_score, feature, kto_tag, idx= predict_results.predictions

        chosen_score = torch.tensor(chosen_score)
        feature = torch.tensor(feature)
        false_indices = torch.nonzero(torch.tensor(kto_tag == False)).squeeze()
        outlier_pred1 = self.find_outlier_with_features(feature[false_indices])
        outlier_pred2 = self.find_outlier_with_scores(chosen_score[false_indices], ratio=0.2)
        comb_results = outlier_pred2#outlier_pred1 * outlier_pred2
        
        final_retults = torch.zeros(len(kto_tag))
        final_retults[false_indices] = comb_results
        logger.info(f"The ratio of selected outliers: " + str(sum(final_retults) / len(final_retults)))
        logger.info(f"The ratio of selected outliers in wrong solutions: " + str(sum(final_retults) / len(false_indices)))

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, tag, outlier_pred, ii in zip(chosen_score.numpy(), kto_tag, final_retults.numpy(), idx):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "tag": round(float(tag), 2), "outlier_tag": bool(outlier_pred), "idx": int(ii)}))

            writer.write("\n".join(res))

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
        
    
        

