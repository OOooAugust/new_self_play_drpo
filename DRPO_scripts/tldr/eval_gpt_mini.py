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



system_prompt = """Which of the following summaries does a better job of summarizing the post? 
Strictly follow these criteria when selecting the best summary: 
1. Prioritize the summary which eliminates unnecessary details and keeps only the author’s main concern or question. 
2. Avoid lengthy sentences, minor details or redundant information, express the key idea in few words. 
3. Prioritize the shorter summary as long as it remains clear and preserves the main idea.  
Post: {prompt}. 
Summary 0: {response0}, Summary 1: {response1}, 
state only "0" or "1" to indicate your choice."""

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
            "judege_reason": reasons,
        })
        
        win_rate = sum(results) / len(results)
        
        temp_str = str(temp).replace(".", "_")
        df.to_csv(f"{method_name}_temp_{temp_str}_results_2.csv", index=False)
        
        return win_rate

data = load_dataset("Eehan/eval-tldr-dpo-drpo-0.75tmp-sft-ppo-1000")
temperatures = [0, 0.25, 0.5, 0.75, 1.0]
all_indices = list(range(3000)) 
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
    
    data_sft_dpo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["dpo"]] for x in temp_data]}
    data_dpo_drpo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo-0.9tmp"]] for x in temp_data]}
    data_sft_drpo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo-0.9tmp"]] for x in temp_data]}
    

    sft_dpo_win_rate = evaluate_and_save(data_sft_dpo, "sft_dpo", temp)
    print(f"sft_dpo_finish temp {temp}")
    dpo_drpo_win_rate = evaluate_and_save(data_dpo_drpo, "dpo_drpo", temp)
    print(f"dpo_drpo_finish temp {temp}")
    sft_drpo_win_rate = evaluate_and_save(data_sft_drpo, "sft_drpo", temp)
    print(f"sft_drpo_finish temp {temp}")


    temp_results = [
        {"Temperature": temp, "Model": "SFT_DPO", "Win Rate": sft_dpo_win_rate},
        {"Temperature": temp, "Model": "DPO_DR-DPO", "Win Rate": dpo_drpo_win_rate},
        {"Temperature": temp, "Model": "SFT_DR-DPO", "Win Rate": sft_drpo_win_rate}
    ]
    
    results_df = pd.concat([results_df, pd.DataFrame(temp_results)], ignore_index=True)

results_df.to_csv("head_to_head_summary_results_1.csv", index=False)
print("Clear by Spring")
