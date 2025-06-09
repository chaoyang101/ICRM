import numpy as np
import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import json
import tqdm

class ATORM(AutoModelForCausalLMWithValueHead):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        config = pretrained_model.config
        hidden_size = config.hidden_size
        self.first_head = nn.Linear(hidden_size, config.vocab_size)
        self.second_head = nn.Linear(hidden_size, config.vocab_size)
        #self._copy_head_weights()


    def _copy_head_weights(self):
        original_head = self.pretrained_model.get_output_embeddings()   
        if original_head is not None:
            assert original_head.weight.shape[0] == self.second_head.weight.shape[0]
            self.second_head.weight = nn.Parameter(original_head.weight.detach().clone())
            if original_head.bias is not None:
                self.second_head.bias = nn.Parameter(original_head.bias.detach().clone())
            else:
                self.second_head.bias = None
        else:
            raise

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
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

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        if hasattr(self.v_head.summary, "weight") and last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, "weight") and (
            last_hidden_state.device != self.v_head.summary[0].weight.device
        ):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)

        # use the last token value as reward
        if torch.any(attention_mask[:, 0] == 0):
            # left padding
            last_index = attention_mask.shape[-1] - 1
        else:
            # right padding
            last_index = attention_mask.sum(dim=-1) - 1
        #value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]
        #value = backward_exp_decay_sum_k_padded_vectorized(self.v_head(last_hidden_state).squeeze(-1), 0.5, 5)[torch.arange(len(last_hidden_state)), last_index]
        return self.v_head(last_hidden_state).squeeze(-1)
    
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
        return pretrained_model_state_dict

if __name__ == "__main__":
    model_path = "./models/40k_fst"
    prompt = "What is an effective way to deal with people whodisagree with me?"
    chosen = "An effective way to deal with people who disagree with you is to respect their view and use kind words."
    rejected = "An effective way to deal with people who disagree with you is to ignore their view and use cruel words."

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ATORM.from_pretrained(model_path).cuda()
    v_head_state_dict = torch.load(os.path.join(model_path, "v_head.pth"))
    model.v_head.load_state_dict(v_head_state_dict)
    model.eval()

    with torch.no_grad():
        messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen},
                    ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        assistant_tokens = tokenizer.encode(chosen, add_special_tokens=False)
        print(assistant_tokens)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        print(model_inputs["input_ids"])
        reward = model(**model_inputs)[0]
        #print(reward)
        print(reward[-2-len(assistant_tokens):-2])

        messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected},
                    ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        print(model_inputs["input_ids"])
        reward = model(**model_inputs)[0]
        #print(reward)
        print(reward[-2-len(assistant_tokens):-2])