import re
import torch
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from trl.data_utils import apply_chat_template


import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

METHOD_NAME = "drpo-0066004"
MODEL_NAME = "Eehan/pythia-1b-deduped-tldr-drpo-base-1e-temp0.66-beta-0.04"  
# METHOD_NAME = "dpo"
# MODEL_NAME = "Kyleyee/pythia-1b-deduped-tldr-dpo"

OUTPUT_DATASET_NAME = "Eehan/eval-tldr"
INPUT_DATASET_NAME = "Kyleyee/train_data_tldr_for_drpo"  
INPUT_DATASET_SPLIT = "validation"  
DATASET_NEED_MERGE = "Eehan/eval-tldr"
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1]  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
print(tokenizer.special_tokens_map)
def generate_text(prompts, tokenizer, model, temperature):
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    generate_kwargs = {
        "max_new_tokens": 64,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    
    outputs = model.generate(
        **inputs,
        **generate_kwargs
    )
    
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def extract_dialogue(examples: dict, tokenizer, model, temperature: float) -> dict:
    prompts = examples["prompt"]
    responses = generate_text(prompts, tokenizer, model, temperature)
    examples[METHOD_NAME] = responses
    return examples

def deduplicate_consecutive_dataset(dataset):
    """对连续重复的prompt进行去重"""
    if len(dataset) <= 1:
        return dataset
        
    unique_indices = [0]  # 第一个元素一定保留
    prev_prompt = dataset[0]['prompt']
    
    # 只需要和前一个prompt比较
    for i in range(1, len(dataset)):
        current_prompt = dataset[i]['prompt']
        if current_prompt != prev_prompt:
            unique_indices.append(i)
            prev_prompt = current_prompt
    
    return dataset.select(unique_indices)

if __name__ == "__main__":

    # dataset = load_dataset(INPUT_DATASET_NAME)[INPUT_DATASET_SPLIT]
    # dataset = deduplicate_consecutive_dataset(dataset).shuffle(seed=36).select(range(3000))
    # dataset = dataset.remove_columns(["rank", "a1", "a2"])

    dataset_merge = load_dataset(DATASET_NEED_MERGE)
    # dataset_merge = DatasetDict()

    

    for temp in TEMPERATURES:
        processed_shard = dataset_merge[f"temperature_{temp}"].map(
            lambda examples: extract_dialogue(examples, tokenizer, model, temp),
            batched=True,
            batch_size=100
        )

        # dataset_merge[f"temperature_{temp}"] = dataset

        dataset_merge[f"temperature_{temp}"] = processed_shard

    dataset_merge.push_to_hub(OUTPUT_DATASET_NAME)
