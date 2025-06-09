import json
import os
from grader import grade_answer
import re

json_file_path = "./data/special tokens.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)
_START = data["input"]["start step"]
_END = data["input"]["end step"]
_REQ = data["input"]["request"]
_SPACE = data["input"]["space holder"]
#_POS = data["output"]["postive"]
#_NEG = data["output"]["negtive"]
pos_key = "postive"
neg_key = "negtive"
nat_key = "natural"

def merge_elements_with_colon(input_list):
    result = []
    i = 0

    while i < len(input_list):
        if input_list[i].endswith(":"):
            if i + 1 < len(input_list): 
                merged_element = input_list[i] + input_list[i + 1]
                result.append(merged_element)
                i += 1 
            else:
                result.append(input_list[i])  
        else:
            result.append(input_list[i]) 

        i += 1  

    return result

def handle_format(question, solution, label, generation_fromat):
    if label:
        step_val = pos_key
    else:
        step_val = neg_key
    if generation_fromat:
        outs = {
            "messages": [
                {
                "content": question,
                "role": "user"
                },
                {
                "content": solution,
                "role": "assistant"
                },
                {
                "content": "Is the answer correct (Yes/No)?",
                "role": "user"
                },
                {
                "content": "Yes" if label else "No", #simplify selection score
                "role": "assistant"
                }
            ],
            "label": label,
            "step_val":step_val
        }
    else:
        outs = {
            "messages": [
                {
                "content": question,
                "role": "user"
                },
                {
                "content": solution,
                "role": "assistant"
                }
            ],
            "label": label,
            "step_val":step_val
        }
    return  outs

def handle2orm(goal_name, file_paths, pass_unsovled=True, generation_fromat=False, filter_func=None):
    if filter_func is None:
        filter_func = lambda x: False
    write_datas = []
    for path in file_paths:
        with open(path, 'r') as json_file:
            read_datas = json.load(json_file)
        bs = len(read_datas[0]["solutions"])
        for line in read_datas:
            if filter_func(line):
                continue
            entry = line
            question = entry["question"]
            inner_write_datas = []
            for solution in entry["solutions"]:
                label = None
                final_response = None
                match_flag = False
                boxed_match = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', solution["steps"][-1])
                for final_response in boxed_match:
                    if grade_answer(final_response, entry["answer"]):
                        label = True
                        match_flag = True
                        break
                if not match_flag and len(boxed_match) > 0:
                    label = False
                if label is None and pass_unsovled:
                    continue
                elif label is None:
                    label = False
                process_solution = solution["steps"]
                process_solution = merge_elements_with_colon(solution["steps"])
                merge_solution = ""
                for step in process_solution:
                    merge_solution += _START + step + _END
                outs = handle_format(question, merge_solution, label, generation_fromat)
                inner_write_datas.append(outs)
            if 0 < len(inner_write_datas) < bs:
                inner_write_datas += [inner_write_datas[-1] for _ in range(bs - len(inner_write_datas))]
            write_datas += inner_write_datas
    print(len(write_datas))

    with open(goal_name, 'w') as json_file:
        json.dump(write_datas, json_file, indent=2)

def filter_consistency(data):
    correct = 0
    max_correct = len(data["solutions"])
    for solution in data["solutions"]:
        label = None
        match_flag = False
        boxed_match = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', solution["steps"][-1])
        for final_response in boxed_match:
            if grade_answer(final_response, data["answer"]):
                label = True
                match_flag = True
                break
        if not match_flag and len(boxed_match) > 0:
            label = False
        if label is None:
            max_correct -= 1
        if label == True:
            correct += 1
    if correct == 0 or correct == max_correct:
        return True
    return False

if __name__ == "__main__":
    goal_name = "./data/train/qwen/qwen_train.json"
    file_paths = ["./data/qwen_answer_generator/generation_train_new4_0_10000.json"]
    handle2orm(goal_name, file_paths, pass_unsovled=False, generation_fromat=False, filter_func=None)
    