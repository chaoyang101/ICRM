import json
def remove_step_from_strings(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.replace("[STEP]", "")
            elif isinstance(value, (dict, list)):
                remove_step_from_strings(value)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, str):
                data[index] = item.replace("[STEP]", "")
            elif isinstance(item, (dict, list)):
                remove_step_from_strings(item)

def process_json_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    remove_step_from_strings(data)
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

input_file_path = "./data/train/qwen/qwen_train.json"  
output_file_path = "./data/train/qwen/qwen_train.json" 
process_json_file(input_file_path, output_file_path)