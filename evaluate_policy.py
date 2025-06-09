import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import tqdm


def score_with_model(predict_results_path, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    scores = 0
    best_score = -10000
    worst_score = 100000
    with open(predict_results_path, 'r', encoding='utf-8') as f:
        eval_datas = json.load(f)
    with torch.no_grad():
        for data in tqdm.tqdm(eval_datas):
            prompt = data["question"]["content"]
            response = data["responses"]["content"]

            if "nicolinho" in model_path:
                messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                output = model(input_ids)
                # Expectation of the reward distribution
                reward = output.score.cpu().float() 
            else:
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
            if reward[0][0] > best_score:
                best_score = reward[0][0]
            if reward[0][0] < worst_score:
                worst_score = reward[0][0]
            scores += reward
    print(worst_score)
    print(best_score)
    print(scores / len(eval_datas))


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
            response1 = data1["responses"]["content"]
            prompt2 = data2["question"]["content"]
            response2 = data2["responses"]["content"]

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


def generate_responses(eval_file_path, model_name, write_file):
    with open(eval_file_path, "r", encoding="utf-8") as f:
        train_inputs = json.load(f)
    #train_inputs = [val["question"]["content"] for val in train_inputs]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        revision="float16",
    )
    all_outputs = []
    for input_text in tqdm.tqdm(train_inputs):
        #input_text = input_text["question"]["content"]
        message = {
            "role": "user",
            "content": input_text,
        }
        message = tokenizer.apply_chat_template([message], tokenize=False)
        input_ids = tokenizer(message, return_tensors="pt").to("cuda")
        input_length = len(input_ids["input_ids"][0])
        outputs = model.generate(**input_ids, max_new_tokens=1024, do_sample=False)
        outputs = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        sample = {
            "question":
            {
                "role": "user",
                "content": input_text
            },
            "responses":{
                    "role": "assistant",
                    "content": outputs
                } 
            
        }
        all_outputs.append(sample)
        with open(write_file, 'w') as json_file:
            json.dump(all_outputs, json_file, indent=2)
    
if __name__ == "__main__":
    model_name1 = "./models/dpo_taorm400k"
    model_name2 = "./models/dpo_grm400k"
    write_file1 = "dpo_taorm_check_400k_greedy_generation.json"
    write_file2 = "dpo_grm_check_400k_greedy_generation.json"
    eval_file_path = "./eval_data/reward_bench_policy_test.json"
    evaluate_model = "nicolinho/QRM-Llama3.1-8B-v2"

    generate_responses(eval_file_path, model_name1, write_file1)
    generate_responses(eval_file_path, model_name2, write_file2)

    get_win_ratio(write_file1, write_file2, evaluate_model)