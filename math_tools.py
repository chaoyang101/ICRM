import json
with open("eval_data/math-llama3.1-8b-inst-64.json", 'r', encoding='utf-8') as f:
    datas = json.load(f)
idx_data_dict = dict()
merged_datas = []
for data in datas:
    idx = data["idx"]
    if idx not in idx_data_dict.keys():
        idx_data_dict[idx] = []
    idx_data_dict[idx].append(data)

for idx, eles in idx_data_dict.items():
    inner_solutions = []
    for ele in eles:
        inner_solutions.append(
            {
                "steps": ele["steps"],
                "correct": ele["correctness"]
            }
        )
    merged_datas.append(
        {
            "question": ele["prompt"],
            "solutions": inner_solutions
        }
    )
    print(len(inner_solutions))
print(len(merged_datas))
with open("eval_data/merged_math-llama3.1-8b-inst-64.json", 'w') as json_file:
    json.dump(merged_datas, json_file, indent=2)