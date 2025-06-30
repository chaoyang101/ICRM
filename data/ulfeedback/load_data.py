from datasets import load_dataset
import json
import os

def build_dataset_UF(data_path, split='train', size=None, mode=''):
    ds = load_dataset(data_path, 'all', split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '4K':
            ds = ds.select(range(0, len(ds), 200)) 
        elif mode == '10K':
            ds = ds.select(range(0, len(ds), 80)) 
        elif mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '100K':
            ds = ds.select(range(0, len(ds), 8)) 
        elif mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 
        else:
            raise

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()

        assert len(chosen_messages[0]['content']) == len(rejected_messages[0]['content'])
        return {
            "content": chosen_messages[0]['content'],
            "chosen": chosen_messages[1]['content'],
            "reject": rejected_messages[1]['content'],
            "margin": margin
        }
    ds = ds.map(formatting_func, batched=False, num_proc=10)
    return ds

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
        
if __name__ == "__main__":
    data_path = "llm-blender/Unified-Feedback"
    dataset_mode='40K' # 400K
    train_path_name = "./data/train/ulfeedback/ulfeedback_train"+dataset_mode+".json"
    eval_path_name = "./data/test/ulfeedback/ulfeedback_eval"+dataset_mode+".json"
    train_ds = build_dataset_UF(data_path, split='train', mode=dataset_mode)
    eval_ds = build_dataset_UF(data_path, split='val', mode=dataset_mode)
    train_ds = change_to_dpo_format(train_ds["content"], train_ds["chosen"], train_ds["reject"])
    eval_ds = change_to_dpo_format(eval_ds["content"], eval_ds["chosen"], eval_ds["reject"])
    if not os.path.exists("./data/train/ulfeedback"):
        os.makedirs("./data/train/ulfeedback")
    if not os.path.exists("./data/test/ulfeedback"):
        os.makedirs("./data/test/ulfeedback")
    with open(train_path_name, 'w') as json_file:
        json.dump(train_ds, json_file, indent=2)
    with open(eval_path_name, 'w') as json_file:
        json.dump(eval_ds, json_file, indent=2)


    