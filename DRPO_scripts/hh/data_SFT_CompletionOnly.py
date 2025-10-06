import sys
import os

# # Add the parent directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser, AutoTokenizer
from trl.data_utils import is_conversational, maybe_apply_chat_template, maybe_convert_to_chatml, pack_examples
@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/hh-rlhf-helpful-base"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="Kyleyee/train_data_HH_sft_CompletionOnly",
        metadata={"help": "Hugging Face repository ID to push the dataset to."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."}
    )

# Optionally define a model card for the new dataset
model_card = ModelCard("""
---
tags: [trl]
---

# HH-RLHF-Helpful-Base SFT Dataset

This dataset duplicates each sample into two, turning `chosen` and `rejected` into separate examples under the `output` column, while renaming `prompt` to `instruction`.
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    processing_class = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    # Load the original dataset
    dataset = load_dataset("Kyleyee/train_data_Helpful_explicit_prompt")
    print(f"Original dataset size: {len(dataset['train'])}")

    # Duplicate and rename fields: prompt -> instruction; chosen/rejected -> output
    def duplicate_and_rename(examples: dict) -> dict:
        instructions = []
        outputs = []
        for prompt, chosen, rejected in zip(examples['prompt'], examples['chosen'], examples['rejected']):
            # First entry: chosen
            chosen = chosen.replace("<|im_end|>\n", "<|im_end|>")
            rejected = rejected.replace("<|im_end|>\n", "<|im_end|>")
            instructions.append(prompt)
            outputs.append(chosen)
            # Second entry: rejected
            instructions.append(prompt)
            outputs.append(rejected)
        return {
            'instruction': instructions,
            'output': outputs
        }

    # Apply mapping with batching to double the size
    dataset = dataset.map(
    maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class},load_from_cache_file=False)
    
    print(f"New dataset after applying chat template: {dataset['train'][0]}")

    dataset = dataset.map(
        duplicate_and_rename,
        batched=True,
        batch_size=256,
        remove_columns=['prompt', 'chosen', 'rejected'],
        num_proc=script_args.dataset_num_proc,
        load_from_cache_file=False
    )
    print(f"New dataset size: {len(dataset)}")

    # Push to Hub
    if script_args.push_to_hub:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            raise RuntimeError(
                "Hugging Face token not found. Set HF_TOKEN or HUGGINGFACE_TOKEN in the environment before pushing."
            )
        dataset.push_to_hub(script_args.repo_id, token=token)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset", token=token)
        print(f"Pushed new dataset to {script_args.repo_id}")
