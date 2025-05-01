import re
import torch
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from trl.data_utils import apply_chat_template

# 初始化设备和模型
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer_sft = AutoTokenizer.from_pretrained("Kyleyee/Qwen2.5-1.5B-sft-hh-3e")
print("ESO TOKEN",tokenizer_sft.eos_token)
print("ESO TOKEN ID",tokenizer_sft.eos_token_id)
print(tokenizer_sft.special_tokens_map)
tokenizer_sft.padding_side = "left"
tokenizer_sft.eos_token = "<|im_end|>"
model_sft = AutoModelForCausalLM.from_pretrained("Kyleyee/Qwen2.5-1.5B-sft-hh-3e").to(device)
print("ESO TOKEN",tokenizer_sft.eos_token)
print("ESO TOKEN ID",tokenizer_sft.eos_token_id)
print(tokenizer_sft.special_tokens_map)


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

def extract_dialogue(examples: dict, temperature: float) -> dict:

    prompts = examples["prompt"]
    chat_prompts = [apply_chat_template({"prompt" : p}, tokenizer_sft) for p in prompts]
    flat_prompts = [x["prompt"] for x in chat_prompts]
    sft_responses = generate_text(flat_prompts, tokenizer_sft, model_sft, temperature)
  

    
    
    # 生成各模型的响应
    return {
        "prompt_chat": flat_prompts,
        "sft": sft_responses,
    }

if __name__ == "__main__":

    dataset = load_dataset("Kyleyee/train_data_Helpful_explicit_prompt")["train"].select(range(100))
    dataset = dataset.remove_columns(["rejected","chosen"])
    

    processed = DatasetDict()
    temperatures = [0, 0.25, 0.5, 0.75, 1]
    
    for temp in temperatures:
        processed_shard = dataset.map(
            extract_dialogue,
            fn_kwargs={"temperature": temp},
            batched=True,
            batch_size=64
        )
        processed[f"temperature_{temp}"] = processed_shard
    
    
    processed.push_to_hub("Kyleyee/eval_for_sft_quality_hh_old_new_end")