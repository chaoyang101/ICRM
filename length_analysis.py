import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import json
import tqdm
import seaborn as sns


def concat_two_length_analysis_image(length_list, label_list, plot_save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    color_list = ["coral", "tan"]
    
    all_lengths = np.concatenate(length_list)
    global_min, global_max = np.min(all_lengths), np.max(all_lengths)

    bins = np.linspace(global_min, global_max, 20) 
    
    for lengths, label, color in zip(length_list, label_list, color_list):
        mean_len = np.mean(lengths)

        sns.histplot(
            lengths,
            kde=True,
            bins=bins, 
            color=color,
            stat="density",
            alpha=0.6,
            label=label,
            kde_kws={"bw_adjust": 0.5} 
        )

        plt.axvline(mean_len, color=color, linestyle="--", linewidth=2)
        print(mean_len)

    plt.xlim(global_min, global_max)
    plt.tick_params(axis='both', labelsize=30)
    
    plt.xlabel("Length", fontsize=36)
    plt.ylabel("Density", fontsize=36)
    plt.legend(fontsize=30)
        
    plt.savefig(
        plot_save_path,
        #dpi=1080,
        bbox_inches='tight',
        transparent=False
    )

def get_tokenizer_length(tokenizer_name, strings):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    for string in strings:
        model_inputs = tokenizer(string, return_tensors="pt").to("cuda")
        lengths.append(len(model_inputs["input_ids"][0]))
    return lengths

if __name__ == "__main__":
    grm_bon_file_name = "reward_bench_selected4_grm_400k.json"
    taorm_bon_file_name = "reward_bench_selected4_taorm_400k.json"
    plot_path = "./plot/N4.pdf"
    tokenizer_path = "./models/400k_fst"

    with open(grm_bon_file_name, 'r', encoding='utf-8') as f:
        datas_grm = json.load(f)
    grm_chosen_list = []
    for data in datas_grm:
        grm_chosen_list.append(data["response"]["content"])
    grm_lengths = get_tokenizer_length(tokenizer_path, grm_chosen_list)

    with open(taorm_bon_file_name, 'r', encoding='utf-8') as f:
        datas_taorm = json.load(f)
    taorm_chosen_list = []
    for data in datas_taorm:
        taorm_chosen_list.append(data["response"]["content"])
    taorm_lengths = get_tokenizer_length(tokenizer_path, taorm_chosen_list)

    concat_two_length_analysis_image([grm_lengths, taorm_lengths], ["GRM", "Ours"], plot_path)
