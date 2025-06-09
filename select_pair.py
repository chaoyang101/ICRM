import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from trl import AutoModelForCausalLMWithValueHead
import os
import logging
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

class  ATORMPipeline:
    def __init__(self, task, model, tokenizer, state):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.state = state

    def __call__(self, samples, **kwargs):
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, evaluate_state=self.state)
        outputs = torch.tensor(outputs)
        return outputs


def select_min_max_indices(score: torch.Tensor, completeness: list) -> tuple:
    true_count = sum(completeness)
    
    if true_count > 1:
        masked_scores = score[completeness]
        max_val, max_idx_masked = torch.max(masked_scores, dim=0)
        min_val, min_idx_masked = torch.min(masked_scores, dim=0)
        
        true_indices = [i for i, val in enumerate(completeness) if val]
        max_idx = true_indices[max_idx_masked.item()]
        min_idx = true_indices[min_idx_masked.item()]
    else:
        max_val, max_idx = torch.max(score, dim=0)
        min_val, min_idx = torch.min(score, dim=0)
        max_idx = max_idx.item()
        min_idx = min_idx.item()
    
    return max_idx, min_idx

if __name__ == "__main__":
    model_path = "./models/400k_fst"
    input_file_list = ["./eval_data/reward_bench_train_content0_6000_generation.json"]
    goal_name = "./data/train/ulfeedback/dpo_taorm_400k_reward_bench.json"
    goal_name_filter = "./data/train/ulfeedback/dpo_taorm_400k_reward_bench_filter.json"
    state = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ATORM.from_pretrained(model_path).cuda()
    v_head_state_dict = torch.load(os.path.join(model_path, "v_head.pth"))
    model.v_head.load_state_dict(v_head_state_dict)
    model.eval()
    pipeline = ATORMPipeline("", model, tokenizer, state=state)

    json_data = []
    for input_file in input_file_list:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_data += data
    
    content_list, chosen_list, reject_list = [], [], []
    for sample in tqdm.tqdm(json_data):
        content = sample["question"]["content"]
        texts = []
        responses = []
        completeness = []
        rewards = []
        for response in sample["responses"]:
            messages = [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": response["content"]},
                    ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
            texts.append(text)
            responses.append(response["content"])

        scores = pipeline(texts)
        #assert len(scores) == len(sample["responses"])

        #max_idx, min_idx = select_min_max_indices(scores, completeness)
        max_idx = torch.argmax(scores) #rewards.index(max(rewards))
        min_idx = torch.argmin(scores) #rewards.index(min(rewards))#
        
        chosen = responses[max_idx]
        reject = responses[min_idx]
        
        content_list.append(content)
        chosen_list.append(chosen)
        reject_list.append(reject)

    
    dpo_datas = change_to_dpo_format(content_list, chosen_list, reject_list)
    with open(goal_name, 'w') as json_file:
        json.dump(dpo_datas, json_file, indent=2)

    with open("./eval_data/reward_bench_500_content.json", "r", encoding="utf-8") as f:
        test_inputs = json.load(f)
    with open(goal_name, "r", encoding="utf-8") as f:
        train_inputs = json.load(f)
    filter_train_inputs = []
    for inputs in train_inputs:
        if inputs["conversations"][0]["content"] not in test_inputs:
            filter_train_inputs.append(inputs)
    with open(goal_name_filter, 'w') as json_file:
        json.dump(filter_train_inputs, json_file, indent=2)

