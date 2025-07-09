from datasets import load_dataset
import json


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

def split_train_test(prompts, chosens, rejecteds, test_step=10):
    assert len(prompts) == len(chosens) == len(rejecteds)
    
    test_indices = list(range(0, len(prompts), test_step))
    
    train_indices = [i for i in range(len(prompts)) if i not in test_indices]
    
    train_prompts = [prompts[i] for i in train_indices]
    train_chosens = [chosens[i] for i in train_indices]
    train_rejecteds = [rejecteds[i] for i in train_indices]
    
    test_prompts = [prompts[i] for i in test_indices]
    test_chosens = [chosens[i] for i in test_indices]
    test_rejecteds = [rejecteds[i] for i in test_indices]
    
    return (
        (train_prompts, train_chosens, train_rejecteds),
        (test_prompts, test_chosens, test_rejecteds)
    )

if __name__ == "__main__":
    train_path_name = "./data/train/skywork/skywork_train.json"
    eval_path_name = "./data/test/skywork/skywork_eval.json"
    
    if not os.path.exists("./data/train/skywork"):
        os.makedirs("./data/train/skywork")
    if not os.path.exists("./data/test/skywork"):
        os.makedirs("./data/test/skywork")
        
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")
    prompts = []
    chosens = []
    rejecteds = []
    for i in range(len(ds['train'])):
        chosen_pair = ds['train'][i]['chosen']
        rejected_pair = ds['train'][i]['rejected']
        assert chosen_pair[0]['content'] == rejected_pair[0]['content']
        prompts.append(chosen_pair[0]['content'])
        chosens.append(chosen_pair[1]['content'])
        rejecteds.append(rejected_pair[1]['content'])
    
    (train_prompts, train_chosens, train_rejecteds), (test_prompts, test_chosens, test_rejecteds) = split_train_test(prompts, chosens, rejecteds)

    train_ds = change_to_dpo_format(train_prompts, train_chosens, train_rejecteds)
    eval_ds = change_to_dpo_format(test_prompts, test_chosens, test_rejecteds)
    with open(train_path_name, 'w') as json_file:
        json.dump(train_ds, json_file, indent=2)
    with open(eval_path_name, 'w') as json_file:
        json.dump(eval_ds, json_file, indent=2)

    stats_prompts = calculate_length_stats(prompts)
    stats_chosens = calculate_length_stats(chosens)
    stats_rejecteds = calculate_length_stats(rejecteds)
    
