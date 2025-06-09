
import numpy as np
import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel
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

def score_with_model_dpo(predict_results_path, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    scores = 0
    with open(predict_results_path, 'r', encoding='utf-8') as f:
        eval_datas = json.load(f)
    with torch.no_grad():
        for data in tqdm.tqdm(eval_datas):
            prompt = data["conversations"][0]["content"]
            response = data["chosen"]["content"]

            message = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
            ]
            message_template = tokenizer.apply_chat_template(message, tokenize=False)
            # it will look like this: "<s><s> [INST] I'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone.  But I can't do that while I'm at the movie.  Can you help by impersonating me by chat with her? [/INST]Sorry, I'm not comfortable impersonating you in that way.  I'm not willing to behave so dishonestly.  Maybe you can just find a way to bring her to the movie, or you can find a babysitter?</s>"

            kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            tokens = tokenizer.encode_plus(message_template, **kwargs)

            reward_tensor = model(tokens["input_ids"].to(model.device), attention_mask=tokens["attention_mask"].to(model.device)).logits.reshape(-1)
            reward = reward_tensor.cpu().detach().item()
            '''
            messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            output = model(input_ids)
            # Expectation of the reward distribution
            reward = output.score.cpu().float() 
            '''
            scores += reward
    print(scores / len(eval_datas))


def score_with_model(predict_results_path, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    scores = 0
    with open(predict_results_path, 'r', encoding='utf-8') as f:
        eval_datas = json.load(f)
    with torch.no_grad():
        for data in tqdm.tqdm(eval_datas):
            prompt = data["question"]["content"]
            response = data["response"]["content"]
            message = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
            ]
            message_template = tokenizer.apply_chat_template(message, tokenize=False)
            # it will look like this: "<s><s> [INST] I'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone.  But I can't do that while I'm at the movie.  Can you help by impersonating me by chat with her? [/INST]Sorry, I'm not comfortable impersonating you in that way.  I'm not willing to behave so dishonestly.  Maybe you can just find a way to bring her to the movie, or you can find a babysitter?</s>"

            kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            tokens = tokenizer.encode_plus(message_template, **kwargs)

            reward_tensor = model(tokens["input_ids"].to(model.device), attention_mask=tokens["attention_mask"].to(model.device)).logits.reshape(-1)
            reward = reward_tensor.cpu().detach().item()
            '''
            messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            output = model(input_ids)
            # Expectation of the reward distribution
            reward = output.score.cpu().float() 
            '''
            scores += reward
    print(scores / len(eval_datas))

def score_with_answer(predict_results_path):
    accuracy = 0
    with open(predict_results_path, 'r', encoding='utf-8') as f:
        eval_datas = json.load(f)
    for data in tqdm.tqdm(eval_datas):
        sel_state = data["sel_state"]
        if sel_state:
            accuracy += 1
    print(accuracy / len(eval_datas))

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

def get_win_ratio(file1, file2, model_path):
    with open(file1, 'r', encoding='utf-8') as f:
        eval_datas1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        eval_datas2 = json.load(f)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    win_ratio = 0
    tie_ratio = 0
    lose_ratio = 0
    average_score1 = 0
    average_score2 = 0
    with torch.no_grad():
        for data1, data2 in zip(eval_datas1, eval_datas2):
            prompt1 = data1["question"]["content"]
            response1 = data1["response"]["content"]
            prompt2 = data2["question"]["content"]
            response2 = data2["response"]["content"]

            assert len(prompt1) == len(prompt2)

            if "nicolinho" in model_path:
                messages = [{"role": "user", "content": prompt1},
                {"role": "assistant", "content": response1}]
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                output = model(input_ids)
                # Expectation of the reward distribution
                reward1 = output.score.cpu().float() 

                messages = [{"role": "user", "content": prompt2},
                {"role": "assistant", "content": response2}]
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                output = model(input_ids)
                # Expectation of the reward distribution
                reward2 = output.score.cpu().float() 

                average_score1 += reward1
                average_score2 += reward2

                if reward1[0][0] > reward2[0][0]:
                    win_ratio += 1
                elif reward1[0][0] == reward2[0][0]:
                    tie_ratio += 1
                else:
                    lose_ratio += 1
    print(win_ratio / len(eval_datas1))
    print(tie_ratio / len(eval_datas1))
    print(lose_ratio / len(eval_datas1))
    print(average_score1 / len(eval_datas1))
    print(average_score2 / len(eval_datas1))

if __name__ == "__main__":
    model_path1 = "./models/400k_fst"
    model_path2 = "./models/400k"
    goal_name1 = "reward_bench_N16_selected_taorm400k.json"
    goal_name2 = "reward_bench_N16_selected_grm400k.json"
    file_path = "./eval_data/reward_bench_500_content0_500_generation.json"
    N = 16
    extra_file_handle_BON(model_path=model_path1, file_path=file_path, goal_name=goal_name1, N=N, state=None)
    extra_file_handle_BON(model_path=model_path2, file_path=file_path, goal_name=goal_name2, N=N, state=None)

    get_win_ratio(goal_name1, goal_name2, "nicolinho/QRM-Llama3.1-8B-v2")