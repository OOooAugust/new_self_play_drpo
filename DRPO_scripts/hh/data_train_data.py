import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import ModelCard






def common_start(str1: str, str2: str) -> str:
    # Zip the two strings and iterate over them together
    common_chars = []
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            common_chars.append(c1)
        else:
            break
    # Join the common characters and return as a string
    return "".join(common_chars)


def extract_dialogue(example: str) -> list[dict[str, str]]:
    # Extract the prompt, which corresponds to the common start of the chosen and rejected dialogues
    prompt_text = common_start(example["chosen"], example["rejected"])

    # The chosen and rejected may share a common start, so we need to remove the common part
    if not prompt_text.endswith("\n\nAssistant: "):
        # print(f"Warning: The prompt does not end with the expected format. Found: {prompt_text}")
        prompt_text = prompt_text[: prompt_text.rfind("\n\nAssistant: ")] + "\n\nAssistant: "

    # Extract the chosen and rejected lines
    a1_line = example["chosen"][len(prompt_text) :]
    a2_line = example["rejected"][len(prompt_text) :]

    # Remove the generation prompt ("\n\nAssistant: ") from the prompt
    prompt_text = prompt_text[: -len("\n\nAssistant: ")]

    # Split the string at every occurrence of "Human: " or "Assistant: "
    prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)

    # Remove the first element as it's empty
    prompt_lines = prompt_lines[1:]

    prompt = []
    for idx in range(0, len(prompt_lines), 2):
        role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
        content = prompt_lines[idx + 1]
        prompt.append({"role": role, "content": content})

    # Remove the prompt from the chosen and rejected dialogues
    a1 = [{"role": "assistant", "content": a1_line}]
    a2 = [{"role": "assistant", "content": a2_line}]
    return {"prompt": prompt, "a1": a1, "a2": a2, "rank": 1}


model_card = ModelCard("""
---
tags: [trl]
---

# HH-RLHF-Helpful-Base Dataset

## Summary

The HH-RLHF-Helpful-Base dataset is a processed version of [Anthropic's HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset, specifically curated to train models using the [TRL library](https://github.com/huggingface/trl) for preference learning and alignment tasks. It contains pairs of text samples, each labeled as either "chosen" or "rejected," based on human preferences regarding the helpfulness of the responses. This dataset enables models to learn human preferences in generating helpful responses, enhancing their ability to assist users effectively.

Columns:
- `"prompt"`: The user query.
- `"a1"`: A response.
- `"a2"`: Another.
- `"rank"`: The rank of the response, where `1` indicates the first response is preferred and `0` indicates the second response is preferred.

This structure enables models to learn the relationship between user queries and their corresponding responses, enhancing their ability to assist users effectively, used for DRPO.

""")

if __name__ == "__main__":


    dataset = load_dataset("Anthropic/hh-rlhf",  data_dir="helpful-base")
    print(f"Loaded dataset: {dataset['train'][0]}")
    dataset = dataset.map(extract_dialogue, remove_columns=["chosen", "rejected"])
    print(f"Applied chat template: {dataset['train'][0]}")


    dataset.push_to_hub("Kyleyee/train_data_hh_for_drpo",)
    model_card.push_to_hub("Kyleyee/train_data_hh_for_drpo", repo_type="dataset")