import sys
import os
import csv

import yaml

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trl import HfPairwiseJudge, OpenAIPairwiseJudge
from datasets import load_dataset
import pandas as pd
import random
from functools import partial
from threading import Lock
from typing import Optional, Union
from concurrent.futures import TimeoutError
from huggingface_hub import InferenceClient

COMPARISON_PATH = """./gpt_evaluation/tldr/{method_name}_temp_{temp_str}_results_3s.csv"""
SOURCE_DATA = "Eehan/eval-tldr"
SOURCE_LENGTH = 3000
SUMMARY_PATH = f"./gpt_evaluation/tldr/apocrypha_3s.csv"
temperatures = [1, 0.75, 0.5]

# system_prompt = """Which of the following summaries does a better job of summarizing the post? 
# Strictly follow these criteria when selecting the best summary: 
# 1. Prioritize the summary which eliminates unnecessary details and keeps only the author’s main concern or question. 
# 2. Avoid lengthy sentences, minor details or redundant information, express the key idea in few words. 
# 3. Prioritize the shorter summary as long as it remains clear and preserves the main idea.  
# Post: {prompt}. 
# Summary 0: {response0}, Summary 1: {response1}, 
# state only "0" or "1" to indicate your choice."""

# system_prompt = """Which of the following summaries does a better job of 
# summarizing the most important points in the given forum post, 
# without including unimportant or irrelevant details? 
# A good summary is both precise and concise.
# Post: {user_query}
# Summary A: {response_a}
# Summary B: {response_b}
# FIRST provide a one-sentence comparison of the two summaries, 
# explaining which you prefer and why. 
# SECOND, on a new line, state only "A" or "B" to indicate your choice. 
# Your response should use the format:
# Comparison: <one-sentence comparison and explanation>
# Preferred: <"A" or "B">"""

system_prompt = """Which of the following summaries does a better job of summarizing the most
important points in the given forum post?
Post: {user_query}
Summary A: {response_a}
Summary B: {response_b}
FIRST provide a one-short-sentence comparison of the two summaries, explaining which
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
choice. Your response should use the format:
Comparison: <one-sentence comparison and short explanation>
Preferred: <"A" or "B">"""


judge = OpenAIPairwiseJudge(model = "gpt-4o-mini", system_prompt=system_prompt)
def evaluate_and_save(data, method_name, temp):
        results, reasons = judge.judge_with_reason(
            prompts=data["prompt"],
            completions=data["completions"]
        )
        
        df = pd.DataFrame({
            "prompt": data["prompt"],
            "completion_1": [c[0] for c in data["completions"]],
            "completion_2": [c[1] for c in data["completions"]],
            "judge_result": results,
            "judge_reason": reasons,
        })
        
        win_rate = sum(results) / len(results)
        print(f"{method_name}, {temp}: {win_rate}")
        
        temp_str = str(temp).replace(".", "")
        df.to_csv(COMPARISON_PATH.format(method_name=method_name, temp_str=temp_str), index=False, escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
        
        return win_rate


data = load_dataset(SOURCE_DATA)
# print(data["temperature_0"].select(range(5)).to_pandas())
all_indices = list(range(SOURCE_LENGTH))
random.seed(66004)
random.shuffle(all_indices)  
num_samples_per_temp = SOURCE_LENGTH // len(data.keys())
print(num_samples_per_temp)

temp_indices = {
    temp: all_indices[i * num_samples_per_temp: (i + 1) * num_samples_per_temp]
    for i, temp in enumerate(temperatures)
}
temperature_data = {
    temp: [data[f"temperature_{temp}"][i] for i in temp_indices[temp]]
    for temp in temp_indices
}

results_df = pd.DataFrame(columns=["Temperature", "Model", "Win Rate"])
for temp in temperatures:
    temp_data = temperature_data[temp]
    
    
    # data_sft_drpo_gpm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
    # data_dpo_drpo_gpm =  {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
    # data_sft_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo-0066004-new"]] for x in temp_data]}
    # data_dpo_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo-0066004-new"]] for x in temp_data]}
    data_ipo_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ipo"], x["drpo-0066004-new"]] for x in temp_data]}
    data_ipo_drpo_gpm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ipo"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
                         
    
    # sft_drpo_gpm_win_rate = evaluate_and_save(data_sft_drpo_gpm, "sft_drpo-gpm", temp)
    # dpo_drpo_gpm_win_rate = evaluate_and_save(data_dpo_drpo_gpm, "dpo-drpo_gpm", temp)
    # sft_drpo_bt_win_rate = evaluate_and_save(data_sft_drpo_bt, "sft-drpo_bt", temp)
    # dpo_drpo_bt_win_rate = evaluate_and_save(data_dpo_drpo_bt, "dpo-drpo_bt", temp)
    ipo_drpo_bt_win_rate = evaluate_and_save(data_ipo_drpo_bt, "ipo-drpo_bt", temp)
    ipo_drpo_gpm_win_rate = evaluate_and_save(data_ipo_drpo_gpm, "ipo-drpo_gpm", temp)


    temp_results = [
        # {"Temperature": temp, "Model": "SFT_DRPO-GPM", "Win Rate": sft_drpo_gpm_win_rate},
        # {"Temperature": temp, "Model": "DPO_DRPO-GPM", "Win Rate": dpo_drpo_gpm_win_rate},
        # {"Temperature": temp, "Model": "SFT_DRPO-BT", "Win Rate": sft_drpo_bt_win_rate},
        # {"Temperature": temp, "Model": "DPO_DRPO-BT", "Win Rate": dpo_drpo_bt_win_rate},
        {"Temperature": temp, "Model": "IPO_DRPO-BT", "Win Rate": ipo_drpo_bt_win_rate},
        {"Temperature": temp, "Model": "IPO_DRPO-GPM", "Win Rate": ipo_drpo_gpm_win_rate},
    ]
    
    results_df = pd.concat([results_df, pd.DataFrame(temp_results)], ignore_index=True)


# results_df.to_csv(SUMMARY_PATH, index=False)
results_df.to_csv(SUMMARY_PATH, mode='a', header=False, index=False)
print("Clear by Spring")