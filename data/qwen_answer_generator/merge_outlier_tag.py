import json

data_file_path = "./data/train/qwen/mix_G.json"
tag_file_path = "./data/train/qwen/outlier_predictions2.jsonl"
merge_file_path = "./data/train/qwen/merge_mix_G2.json"

with open(data_file_path, 'r') as file:
    datas = json.load(file)

tags = []
kto_tags = []
idxs = []
idx_shift = 0
last_idx = 0
with open(tag_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        if entry["idx"] < last_idx:
            idx_shift += 1
        last_idx = entry["idx"]
        kto_tags.append(bool(entry["tag"]))
        tags.append(entry["outlier_tag"])
        idxs.append(entry["idx"])
assert len(tags) == len(datas)

merge_datas = []
for data, tag, kto_tag in zip(datas, tags, kto_tags):
    assert data["label"] == kto_tag
    data["step_val"] = tag
    merge_datas.append(data)

with open(merge_file_path, 'w') as json_file:
    json.dump(merge_datas, json_file, indent=2)

