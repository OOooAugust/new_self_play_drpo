import sys
import os

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



# system_prompt = """Which of the following summaries does a better job of summarizing the post? 
# Strictly follow these criteria when selecting the best summary: 
# 1. Prioritize the summary which eliminates unnecessary details and keeps only the author’s main concern or question. 
# 2. Avoid lengthy sentences, minor details or redundant information, express the key idea in few words. 
# 3. Prioritize the shorter summary as long as it remains clear and preserves the main idea.  
# Post: {prompt}. 
# Summary 0: {response0}, Summary 1: {response1}, 
# state only "0" or "1" to indicate your choice."""

system_prompt = """Which of the following summaries does a better job of 
summarizing the most important points in the given forum post, 
without including unimportant or irrelevant details? 
A good summary is both precise and concise.
Post: {user_query}
Summary A: {response_a}
Summary B: {response_b}
FIRST provide a one-sentence comparison of the two summaries, 
explaining which you prefer and why. 
SECOND, on a new line, state only "A" or "B" to indicate your choice. 
Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">"""

# system_prompt = """Which of the following summaries does a better job of summarizing the most
# important points in the given forum post?
# Post: {user_query}
# Summary A: {response_a}
# Summary B: {response_b}
# FIRST provide a one-short-sentence comparison of the two summaries, explaining which
# you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
# choice. Your response should use the format:
# Comparison: <one-sentence comparison and short explanation>
# Preferred: <"A" or "B">"""


judge = OpenAIPairwiseJudge(model = "gpt-4o-mini", system_prompt=system_prompt)
def evaluate_and_save(data, method_name, temp):
        results, reasons = judge.judge_with_reason(
        results, reasons = judge.judge_with_reason(
            prompts=data["prompt"],
            completions=data["completions"]
        )
        
        df = pd.DataFrame({
            "prompt": data["prompt"],
            "completion_1": [c[0] for c in data["completions"]],
            "completion_2": [c[1] for c in data["completions"]],
            "judge_result": results,
            "judege_reason": reasons,
            "judge_result": results,
            "judege_reason": reasons,
        })
        
        win_rate = sum(results) / len(results)
        
        temp_str = str(temp).replace(".", "_")
        df.to_csv(f"{method_name}_temp_{temp_str}_results_6.csv", index=False)
        
        return win_rate


from datasets import DatasetDict

def merge_datasets_by_prompt(dataset1, dataset2, join_on="prompt"):
    """
    Merge two DatasetDicts by joining on a common column (e.g., 'prompt') for each split.
    Assumes:
      - dataset1 and dataset2 are DatasetDicts
      - each split (like 'train', 'test') exists in both datasets
    """
    merged_splits = {}

    for split in dataset1.keys():
        ds1 = dataset1[split]
        ds2 = dataset2[split]

        # Convert to pandas for easier merging
        df1 = ds1.to_pandas()
        df2 = ds2.to_pandas()

        # Merge on the 'prompt' column
        merged_df = df1.merge(df2, on=join_on, how='left')

        # Convert back to Huggingface Dataset
        from datasets import Dataset
        merged_ds = Dataset.from_pandas(merged_df)

        merged_splits[split] = merged_ds

    return DatasetDict(merged_splits)


data = load_dataset("Eehan/eval-tldr-dpo-ppo-drpo-dm-sft-1000-cut")
data2 = load_dataset("Eehan/eval-tldr-dpo-ppo-drpo-dm-sft-1000-cut2")

data = merge_datasets_by_prompt(data, data2)
print(data["temperature_0"].select(range(5)).to_pandas())


temperatures = [0, 0.25, 0.5, 0.75, 1.0]
all_indices = list(range(1000))
random.shuffle(all_indices)  
num_samples_per_temp = len(all_indices) // len(data.keys())
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
    
    # data_sft_dpo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["dpo"]] for x in temp_data]}
    # data_sft_ppo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["ppo"]] for x in temp_data]}
    # data_sft_drpo_lowtemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo_lowtemp"]] for x in temp_data]}
    # data_sft_drpo_hightemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo_hightemp"]] for x in temp_data]}
    # data_sft_dm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["dm"]] for x in temp_data]}
    data_sft_drpo_medtemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo_medtemp"]] for x in temp_data]}
    data_sft_drpo_smallbeta = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo_smallbeta"]] for x in temp_data]}

    # data_dpo_drpo_lowtemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo_lowtemp"]] for x in temp_data]}
    # data_dpo_drpo_hightemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo_hightemp"]] for x in temp_data]}
    # data_dpo_dm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["dm"]] for x in temp_data]}

    # data_ppo_drpo_lowtemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo_lowtemp"]] for x in temp_data]}
    # data_ppo_drpo_hightemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo_hightemp"]] for x in temp_data]}
    # data_ppo_dm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["dm"]] for x in temp_data]}
    data_ppo_drpo_medtemp = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo_medtemp"]] for x in temp_data]}
    data_ppo_drpo_smallbeta = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo_smallbeta"]] for x in temp_data]}
    

    # sft_dpo_win_rate = evaluate_and_save(data_sft_dpo, "sft_dpo", temp)
    # sft_ppo_win_rate = evaluate_and_save(data_sft_ppo, "sft_ppo", temp)
    # sft_drpo_lowtemp_win_rate = evaluate_and_save(data_sft_drpo_lowtemp, "sft_drpo_lowtemp", temp)
    # sft_drpo_hightemp_win_rate = evaluate_and_save(data_sft_drpo_hightemp, "sft_drpo_hightemp", temp)
    # sft_dm_win_rate = evaluate_and_save(data_sft_dm, "sft_dm", temp)
    sft_drpo_medtemp_win_rate = evaluate_and_save(data_sft_drpo_medtemp, "sft_drpo_medtemp", temp)
    sft_drpo_smallbeta_win_rate = evaluate_and_save(data_sft_drpo_smallbeta, "sft_drpo_smallbeta", temp)

    # dpo_drpo_lowtemp_win_rate = evaluate_and_save(data_dpo_drpo_lowtemp, "dpo_drpo_lowtemp", temp)
    # dpo_drpo_hightemp_win_rate = evaluate_and_save(data_dpo_drpo_hightemp, "dpo_drpo_hightemp", temp)
    # dpo_dm_win_rate = evaluate_and_save(data_dpo_dm, "dpo_dm", temp)

    # ppo_drpo_lowtemp_win_rate = evaluate_and_save(data_ppo_drpo_lowtemp, "ppo_drpo_lowtemp", temp)
    # ppo_drpo_hightemp_win_rate = evaluate_and_save(data_ppo_drpo_hightemp, "ppo_drpo_hightemp", temp)
    # ppo_dm_win_rate = evaluate_and_save(data_ppo_dm, "ppo_dm", temp)
    ppo_drpo_medtemp_win_rate = evaluate_and_save(data_ppo_drpo_medtemp, "ppo_drpo_medtemp", temp)
    ppo_drpo_smallbeta_win_rate = evaluate_and_save(data_ppo_drpo_smallbeta, "ppo_drpo_smallbeta", temp)


    temp_results = [
        # {"Temperature": temp, "Model": "SFT_DPO", "Win Rate": sft_dpo_win_rate},
        # {"Temperature": temp, "Model": "SFT_PPO", "Win Rate": sft_ppo_win_rate},
        # {"Temperature": temp, "Model": "SFT_DRPO_LOWTEMP", "Win Rate": sft_drpo_lowtemp_win_rate},
        # {"Temperature": temp, "Model": "SFT_DRPO_HIGHTEMP", "Win Rate": sft_drpo_hightemp_win_rate},
        # {"Temperature": temp, "Model": "SFT_DM", "Win Rate": sft_dm_win_rate},
        {"Temperature": temp, "Model": "SFT_DRPO_MEDTEMP", "Win Rate": sft_drpo_medtemp_win_rate},
        {"Temperature": temp, "Model": "SFT_DRPO_SMALLBETA", "Win Rate": sft_drpo_smallbeta_win_rate},

        # {"Temperature": temp, "Model": "DPO_DRPO_LOWTEMP", "Win Rate": dpo_drpo_lowtemp_win_rate},
        # {"Temperature": temp, "Model": "DPO_DRPO_HIGHTEMP", "Win Rate": dpo_drpo_hightemp_win_rate},
        # {"Temperature": temp, "Model": "DPO_DM", "Win Rate": dpo_dm_win_rate},

        # {"Temperature": temp, "Model": "PPO_DRPO_LOWTEMP", "Win Rate": ppo_drpo_lowtemp_win_rate},
        # {"Temperature": temp, "Model": "PPO_DRPO_HIGHTEMP", "Win Rate": ppo_drpo_hightemp_win_rate},
        # {"Temperature": temp, "Model": "PPO_DM", "Win Rate": ppo_dm_win_rate},
        {"Temperature": temp, "Model": "PPO_DRPO_MEDTEMP", "Win Rate": ppo_drpo_medtemp_win_rate},
        {"Temperature": temp, "Model": "PPO_DRPO_SMALLBETA", "Win Rate": ppo_drpo_smallbeta_win_rate},  
    ]
    
    results_df = pd.concat([results_df, pd.DataFrame(temp_results)], ignore_index=True)

results_df.to_csv("head_to_head_summary_results_6.csv", index=False)
print("Clear by Spring")

