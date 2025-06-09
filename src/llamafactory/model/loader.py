# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import deepspeed

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


from torch.autograd import Function
from typing import Tuple

class GradientDecayFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output * ctx.coeff, None
    
class reAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        config = pretrained_model.config
        hidden_size = config.hidden_size
        self.base_head = nn.Linear(hidden_size, config.vocab_size)
        self.second_head = nn.Linear(hidden_size, config.vocab_size)
        self.first_head = nn.Linear(hidden_size, config.vocab_size)
        #self.third_head = nn.Linear(hidden_size, config.vocab_size)
        self._copy_head_weights()
        


    def _copy_head_weights(self):
        original_head = self.pretrained_model.get_output_embeddings()  
        with deepspeed.zero.GatheredParameters(original_head.parameters()):
            if original_head is not None:
                assert original_head.weight.shape[0] == self.second_head.weight.shape[0]
                self.first_head.weight = nn.Parameter(original_head.weight.detach().clone())
                self.second_head.weight = nn.Parameter(original_head.weight.detach().clone())
                #self.third_head.weight = nn.Parameter(original_head.weight.detach().clone())
                self.base_head.weight = nn.Parameter(original_head.weight.detach().clone())
                try:
                    if original_head.bias is not None:
                        self.second_head.bias = nn.Parameter(original_head.bias.clone())
                        self.first_head.bias = nn.Parameter(original_head.bias.clone())
                        #self.third_head.bias = nn.Parameter(original_head.bias.clone())
                        self.base_head.bias = nn.Parameter(original_head.bias.clone())
                    else:
                        self.second_head.bias = None
                        self.first_head.bias = None
                        #self.third_head.bias = None
                        self.base_head.bias = None
                except:
                    self.second_head.bias = None
                    self.first_head.bias = None
                    self.base_head.bias = None
                    #self.third_head.bias = None
            else:
                raise

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        return_last_hidden_states=False,
        return_dual_logits=False,
        pos_gradient_alpha=0.001,
        neg_gradient_alpha=0,
        train_sft=-1,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")


        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)
        #sub_value = self.sub_v_head(last_hidden_state).squeeze(-1)


        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        if return_dual_logits:
            if pos_gradient_alpha == 0:
                base_logits = self.base_head(GradientDecayFunction.apply(last_hidden_state.to(self.base_head.weight.dtype), pos_gradient_alpha))
                lm_logits = (base_logits, None, None)
            else:
                if train_sft == 2: # fst
                    second_head_logits = self.second_head(last_hidden_state)
                    base_logits = self.second_head(last_hidden_state)#
                else:
                    second_head_logits = self.base_head(last_hidden_state.detach())#self.second_head(last_hidden_state.detach())#(GradientDecayFunction.apply(last_hidden_state.to(self.second_head.weight.dtype), neg_gradient_alpha))
                    base_logits = self.first_head(GradientDecayFunction.apply(last_hidden_state.to(self.first_head.weight.dtype), pos_gradient_alpha))#(last_hidden_state)
                lm_logits = (base_logits, second_head_logits)

        if return_last_hidden_states:
            if return_past_key_values:
                return (lm_logits, loss, value, last_hidden_state, base_model_output.past_key_values)
            else:
                return (lm_logits, loss, value, last_hidden_state)
        if return_past_key_values:
            return (lm_logits, loss, value, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value)
    
    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        second_head_state_dict = self.second_head.state_dict(*args, **kwargs)
        for k, v in second_head_state_dict.items():
            pretrained_model_state_dict[f"second_head.{k}"] = v
        first_head_state_dict = self.first_head.state_dict(*args, **kwargs)
        for k, v in first_head_state_dict.items():
            pretrained_model_state_dict[f"first_head.{k}"] = v
        base_head_state_dict = self.base_head.state_dict(*args, **kwargs)
        for k, v in base_head_state_dict.items():
            pretrained_model_state_dict[f"base_head.{k}"] = v
        return pretrained_model_state_dict

def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.warning("Processor was not found: {}.".format(e))
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM
            if model_args.train_from_scratch:
                model = load_class.from_config(config)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = reAutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path
        #print(model.second_head.weight.data)
        with deepspeed.zero.GatheredParameters(model.parameters()):
            vhead_params = load_valuehead_params(vhead_path, model_args)
            if vhead_params is not None:
                model.load_state_dict(vhead_params, strict=False)
                logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))
        #print(model.second_head.weight.data)
        #raise

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
