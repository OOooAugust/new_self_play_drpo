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

METHOD_NAME = "drpo-asymtemp"
MODEL_NAME = "Eehan/Qwen2.5-1.5B-drpo-hh-1.5e-temp-7525-beta-0.05"  
OUTPUT_DATASET_NAME = "Eehan/eval-hh"
INPUT_DATASET_NAME = "Kyleyee/train_data_Helpful_explicit_prompt"  
INPUT_DATASET_SPLIT = "test"  
DATASET_NEED_MERGE = "Eehan/eval-hh"
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1]  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
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
        "max_new_tokens": 128,
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
    chat_prompts = [apply_chat_template({"prompt": p}, tokenizer) for p in prompts]
    flat_prompts = [x["prompt"] for x in chat_prompts]
    responses = generate_text(flat_prompts, tokenizer, model, temperature)
    
    return {
        "generated_response": responses
    }

if __name__ == "__main__":

    dataset = load_dataset(INPUT_DATASET_NAME)[INPUT_DATASET_SPLIT]
    dataset_merge = load_dataset(DATASET_NEED_MERGE)
    dataset = dataset.remove_columns(["rejected", "chosen"])
    

    for temp in TEMPERATURES:
        processed_shard = dataset.map(
            lambda examples: extract_dialogue(examples, tokenizer, model, temp),
            batched=True,
            batch_size=64
        )
        dataset_merge[f"temperature_{temp}"] = dataset_merge[f"temperature_{temp}"].add_column(
            METHOD_NAME,
            processed_shard["generated_response"]
        )


    dataset_merge.push_to_hub(OUTPUT_DATASET_NAME)
