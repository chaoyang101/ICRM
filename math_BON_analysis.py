import matplotlib.pyplot as plt
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

def backward_exp_decay_sum_k_padded_vectorized(
    x: torch.Tensor,  # (B, L)
    alpha: float,
    k: int,
) -> torch.Tensor:
    B, L = x.shape
    device = x.device
    # (2) Backward decay: pad past elements with x[:, :1]
    x_padded_bwd = torch.cat([x[:, :1].expand(B, k), x], dim=1)  # (B, k + L)
    bwd = x.clone()
    for step in range(1, k+1):
        #if (alpha ** step) <= 0.01:
        #    break
        bwd +=  (alpha ** step) * x_padded_bwd[:, k - step : k - step + L]

    y = bwd
    return y

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
        evaluate_state=None,
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
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]
        return value
    
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

def change_to_dpo_format(content_list, chosen_list, reject_list):
    datas = []
    assert len(content_list) == len(chosen_list) and len(chosen_list) == len(reject_list)
    for content, chosen, reject in zip(content_list, chosen_list, reject_list):
        sample = {
            "conversations": [
            {
                "role": "user",
                "content": content
            }
            ],
            "chosen": {
                "role": "assistant",
                "content": chosen
            },
            "rejected": {
                "role": "assistant",
                "content": reject
            }
        }
        datas.append(sample)
    return datas

def extra_file_handle(model_path, file_path, goal_name):
    #model_path = ""
    #file_path = ""
    #goal_name = "predict_"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ATORM.from_pretrained(model_path).cuda()
    v_head_state_dict = torch.load(os.path.join(model_path, "v_head.pth"))
    model.v_head.load_state_dict(v_head_state_dict)
    model.eval()

    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    predict_results = []
    for sample in datas:
        prompt = sample["question"][0]["content"]
        chosen = sample["chosen"]["content"]
        reject = sample["rejected"]["content"]

        messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen},
                    ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            chosen_reward = model(**model_inputs)[0]

        messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": reject},
                    ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        model_inputs = tokenizer([text], return_tensors="pt", evaluate_state=None).to("cuda")
        with torch.no_grad():
            reject_reward = model(**model_inputs)[0]

        if reject_reward > chosen_reward:
            sample["chosen"]["content"] = reject
            sample["rejected"]["content"] = chosen
        out_sample = sample
        predict_results.append(out_sample)
    with open(goal_name, 'w') as json_file:
        json.dump(predict_results, json_file, indent=2)

def extra_file_handle_BON(model_path, file_path, goal_name, N, state=None):
    #model_path = ""
    #file_path = ""
    #goal_name = "predict_"
    #N = 4

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ATORM.from_pretrained(model_path).cuda()
    v_head_state_dict = torch.load(os.path.join(model_path, "v_head.pth"))
    model.v_head.load_state_dict(v_head_state_dict)
    model.eval()

    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    predict_results = []
    for sample in tqdm.tqdm(datas):
        prompt = sample["question"]["content"]
        responses = sample["responses"][:N]
        sel_response = None
        sel_reward = -10000
        with torch.no_grad():
            for i in range(N):
                messages = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": responses[i]["content"]},
                            ]
                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False
                    )
                model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
                reward = model(**model_inputs, evaluate_state=state)[0]
                #if not is_sentence_complete(responses[i]["content"]):
                #    reward -= 1000
                if reward > sel_reward:
                    sel_response = responses[i]["content"]
                    sel_reward = reward
        assert sel_response is not None
        out_sample = sample
        out_sample = {
            "question":
            {
                "role": "user",
                "content": prompt
            },
            "response":{
                "role": "assistant",
                "content": sel_response
            },
        }
        predict_results.append(out_sample)
    with open(goal_name, 'w') as json_file:
        json.dump(predict_results, json_file, indent=2)

def extra_file_handle_BON_math(model_path, file_path, goal_name, N, state=None):
    #model_path = ""
    #file_path = ""
    #goal_name = "predict_"
    #N = 4

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ATORM.from_pretrained(model_path).cuda()
    v_head_state_dict = torch.load(os.path.join(model_path, "v_head.pth"))
    model.v_head.load_state_dict(v_head_state_dict)
    model.eval()

    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    predict_results = []
    for sample in tqdm.tqdm(datas):
        prompt = sample["question"]
        responses = sample["solutions"][:N]
        sel_response = None
        sel_reward = -10000
        sel_state = None
        with torch.no_grad():
            for i in range(N):
                solution = "".join(responses[i]["steps"])
                messages = [
                    #{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": solution}
                ]
                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False
                    )
                model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
                reward = model(**model_inputs, evaluate_state=state)[0]
                if "response" in responses[i].keys():
                    if responses[i]["response"] is None:
                        reward -= 1000
                if reward > sel_reward:
                    sel_response = solution
                    sel_reward = reward
                    sel_state = responses[i]["correct"]
        out_sample = sample
        out_sample = {
            "question":
            {
                "role": "user",
                "content": prompt
            },
            "response":{
                "role": "assistant",
                "content": sel_response
            },
            "sel_state": sel_state
        }
        predict_results.append(out_sample)
    with open(goal_name, 'w') as json_file:
        json.dump(predict_results, json_file, indent=2)

def get_tokenizer_length(tokenizer_name, strings):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    for string in strings:
        model_inputs = tokenizer(string, return_tensors="pt").to("cuda")
        lengths.append(len(model_inputs["input_ids"][0]))
    return lengths

def score_with_answer(predict_results_path):
    accuracy = 0
    with open(predict_results_path, 'r', encoding='utf-8') as f:
        eval_datas = json.load(f)
    for data in tqdm.tqdm(eval_datas):
        sel_state = data["sel_state"]
        if sel_state:
            accuracy += 1
    print(accuracy / len(eval_datas))

if __name__ == "__main__":
    extra_file_handle_BON_math(model_path="./models/taorm_math", file_path="./eval_data/merged_math-llama3.1-8b-inst-64.json", goal_name="eval_data/llama_math_selected2_taorm_math.json", N=2, state=None)
    extra_file_handle_BON_math(model_path="./models/taorm_math", file_path="./eval_data/merged_math-llama3.1-8b-inst-64.json", goal_name="eval_data/llama_math_selected4_taorm_math.json", N=4, state=None)
    extra_file_handle_BON_math(model_path="./models/taorm_math", file_path="./eval_data/merged_math-llama3.1-8b-inst-64.json", goal_name="eval_data/llama_math_selected8_taorm_math.json", N=8, state=None)
    extra_file_handle_BON_math(model_path="./models/taorm_math", file_path="./eval_data/merged_math-llama3.1-8b-inst-64.json", goal_name="eval_data/llama_math_selected16_taorm_math.json", N=16, state=None)

    score_with_answer("eval_data/llama_math_selected2_taorm_math.json")
    score_with_answer("eval_data/llama_math_selected4_taorm_math.json")
    score_with_answer("eval_data/llama_math_selected8_taorm_math.json")
    score_with_answer("eval_data/llama_math_selected16_taorm_math.json")
