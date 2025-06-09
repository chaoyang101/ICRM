from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import copy
import re
import time
import os
import torch
from grader import grade_answer
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_question(path):
    questions = []
    answers = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            question = entry["question"]["problem"]
            answer = entry["question"]["ground_truth_answer"]
            if question not in questions:
                questions.append(question)
                answers.append(answer)
    return questions, answers
'''
def load_question2():
    path = "./data/prm_800k_handler/data/phase2_test.jsonl"
    questions = []
    answers = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            question = entry["question"]["problem"]
            answer = entry["question"]["ground_truth_answer"]
            if question not in questions:
                questions.append(question)
                answers.append(answer)
    return questions, answers
'''
def save_answer():
    pass


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

@torch.no_grad()
def generate(train=True, ranges=[0, 3000]):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" # the device to load the model onto
    generation_num_per_question = 4
    max_new_tokens = 2048

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if train:
        goal_name = "./data/qwen_answer_generator/generation_train_new4_"+str(ranges[0])+"_"+str(ranges[1])+".json"
        questions, answers = load_question("./data/prm_800k_handler/data/phase2_train.jsonl")
        questions = questions[ranges[0]:ranges[1]]
        answers = answers[ranges[0]:ranges[1]]
    else:
        goal_name = "./data/qwen_answer_generator/generation_test_whole16_1.5B_2048.json"
        questions, answers = load_question("./data/prm_800k_handler/data/phase2_test.jsonl")
        questions = questions
        answers = answers
    
    #print(len(questions))
    #questions2, answers2 = load_question("./data/prm_800k_handler/data/phase1_train.jsonl")
    #print(len(questions2))
    #print(len(list(set(questions + questions2))))
    generations = []
    ge_format = {"question":"", "answer":"", "solutions":[]}
    so_format = {"steps":[], "response":"", "correct":None}
    num = 0
    assert len(questions) == len(answers)
    for que, ans in tqdm.tqdm(zip(questions, answers), total=len(questions), desc="Processing"):
        #num += 1
        #if num % (len(questions)//10) == 0:
        #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " processed number:" + str(num))
        #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " processed ratio:" + str(100 * num / len(questions)))
        inner_ge = copy.deepcopy(ge_format)
        inner_ge["question"] = que
        inner_ge["answer"] = ans
        inner_solutions = []
        for _ in range(generation_num_per_question):
            prompt = que

            # CoT
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]

            # TIR
            #messages = [
            #    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
            #    {"role": "user", "content": prompt}
            #]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7, 
                top_p=0.8,
                do_sample=True
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            split_response = re.split(r'\n\n', response)
            response_formate = copy.deepcopy(so_format)
            response_formate["steps"] = list(split_response)
            boxed_match = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', split_response[-1])
            match_flag = False
            final_response = None
            for final_response in boxed_match:
                if grade_answer(final_response, ans):
                    response_formate["response"] = final_response
                    response_formate["correct"] = True
                    match_flag = True
                    break
            if not match_flag:
                response_formate["response"] = final_response
                response_formate["correct"] = False
            inner_solutions.append(response_formate)
        inner_ge["solutions"] = inner_solutions
        generations.append(inner_ge)
    
        with open(goal_name, 'w') as json_file:
            json.dump(generations, json_file, indent=2)

@torch.no_grad()
def generate_thinking(train=True):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" # the device to load the model onto
    generation_num_per_question = 4

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if train:
        goal_name = "./data/qwen_answer_generator/generation_train_thinking_7B_temp.json"
        questions, answers = load_question("./data/prm_800k_handler/data/phase1_train.jsonl")
        print(len(questions))
        questions = questions[:10]
        answers = answers[:10]
    else:
        goal_name = "./data/qwen_answer_generator/generation_test_thinking.json"
        questions, answers = load_question("./data/prm_800k_handler/data/phase2_test.jsonl")
    
    #print(len(questions))
    #questions2, answers2 = load_question("./data/prm_800k_handler/data/phase1_train.jsonl")
    #print(len(questions2))
    #print(len(list(set(questions + questions2))))
    generations = []
    ge_format = {"question":"", "answer":"", "solutions":[]}
    num = 0
    for que, ans in zip(questions, answers):
        num += 1
        if num % (len(questions)//10) == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " processed number:" + str(num))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " processed ratio:" + str(100 * num / len(questions)))
        inner_ge = copy.deepcopy(ge_format)
        inner_ge["question"] = que
        inner_ge["answer"] = ans
        inner_solutions = []
        for _ in range(generation_num_per_question):
            prompt = que

            # CoT
            messages = [
                {"role": "system", "content": "You are a highly skilled teacher tasked with guiding a student through the process of solving complex math problems. Please provide a structured outline of thinking for the problem, allowing the student to arrive at the correct answer step by step. Make sure the response is as brief as possible."},
                {"role": "user", "content": prompt}
            ]

            # TIR
            #messages = [
            #    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
            #    {"role": "user", "content": prompt}
            #]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7, 
                top_p=0.8,
                do_sample=True
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            inner_solutions.append(response)
        inner_ge["solutions"] = inner_solutions
        generations.append(inner_ge)
    
        with open(goal_name, 'w') as json_file:
            json.dump(generations, json_file, indent=2)


if __name__ == "__main__":
    generate(train=True, ranges=[0,10000])

