# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import ModelCard
# from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/tldr-preference"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="Kyleyee/train_data_tldr_for_drpo",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_preference(example):
    info = example["info"]
    if example["batch"] in ["batch0_cnndm", "cnndm0", "cnndm2"]:  # CNN Daily Mail batches
        count = 0
        # article = info["article"].replace("\n\n", "\n")
        # prompt = f"TITLE: {info['title']}\n\n{article}\n\nTL;DR:"
    elif example["batch"] in [f"batch{i}" for i in range(3, 23)] + ["edit_b2_eval_test"]:  # Reddit batches
        post = info["post"].replace("\n\n", "\n")
        prompt = f"SUBREDDIT: r/{info['subreddit']}\n\nTITLE: {info['title']}\n\nPOST: {post}\n\nTL;DR:"
        a1 = example["summaries"][0]["text"]
        a2 = example["summaries"][1]["text"]
        rank = example["choice"]
            # chosen_idx = example["choice"]
            # rejected_idx = 1 - chosen_idx
            # chosen = example["summaries"][chosen_idx]["text"]
            # rejected = example["summaries"][rejected_idx]["text"]
        return {"prompt": prompt, "a1": a1, "a2": a2, "rank": 1-rank}
    else:
        raise ValueError(f"Unknown batch: {example['batch']}")




model_card = ModelCard("""
---
tags: [trl]
---

# TL;DR Dataset for DRPO

## Summary

The TL;DR dataset is a processed version of Reddit posts

## Data Structure


Columns:
- `"prompt"`: The unabridged Reddit post.
- `"a1"`: A summary of the post.
- `"a2"`: An alternative summary of the post.
- `"rank"`: The rank of the summary, where `1` indicates the first summary is preferred and `0` indicates the second summary is preferred.


This structure enables models to learn the relationship between detailed content and its abbreviated form, enhancing their summarization capabilities.

""")

if __name__ == "__main__":
    push_to_hub = True

    dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

    dataset = dataset.map(
        to_preference,
        remove_columns=["info", "summaries", "choice", "worker", "batch", "split", "extra"],
    )

    if push_to_hub:
        dataset.push_to_hub("Kyleyee/train_data_tldr_for_drpo")
        model_card.push_to_hub("Kyleyee/train_data_tldr_for_drpo", repo_type="dataset")
