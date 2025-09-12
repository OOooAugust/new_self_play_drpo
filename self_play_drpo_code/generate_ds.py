import os
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
LOCAL_TRL_PARENT = "/workspace/Self_play_DRPO"
if LOCAL_TRL_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_TRL_PARENT)
import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") 
from trl import (
    DPOTrainer,
    DPOConfig,
    ModelConfig,
    DRPOTrainer,
    DRPOConfig,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.data_utils import apply_chat_template
from datasets import load_dataset, concatenate_datasets, DatasetDict
data_cache_path = "/workspace/dataset"
#ultrafeedback_ds = load_dataset('august66/DRPO_data_from_ultrafeed_new_template', split="train", cache_dir=data_cache_path)
hh_ds = load_dataset('Kyleyee/train_data_hh_for_drpo', split = 'train', cache_dir=data_cache_path)


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_path = "/workspace/model_cache"

model_args = ModelConfig(model_name)
model_torch_dtype = torch.bfloat16

lm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    torch_dtype=model_torch_dtype,
    trust_remote_code=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    use_fast=True,
    trust_remote_code=True,
    cache_dir=cache_path,
)

if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


def generate_text(prompts, tokenizer, model, temperature):

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    generate_kwargs = {
        "max_new_tokens": 512,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
        "num_return_sequences": 2,
        'use_cache': True
    }
    
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    
    outputs = model.generate(
        **inputs,
        **generate_kwargs
    )
    
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def get_preference(prompts, response_1, response_2):
    compare_result = blender.compare(prompts, response_1, response_2)
    if compare_result[0]:
        a1 =response_1
        a2 = response_2
    else:
        a2 = response_1
        a1 = response_2
    return a1, a2

def truncate_human(texts):
    return [text.split("\n\nHuman")[0] for text in texts]

def extract_dialogue(examples: dict, tokenizer, model, temperature: float) -> dict:
    # each item is a full chat: list[{"role": "...", "content": "..."}]
    chats = examples["prompt"]

    # render each chat to a single prompt string (all rounds kept)
    rendered = [
        tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,   # append assistant prefix for generation
            tokenize=False
        )
        for chat in chats
    ]

    responses = generate_text(rendered, tokenizer, model, temperature)
    responses = truncate_human(responses)
    return {"generated_response": responses}

def prepare_dataset(batch, tokenizer=tokenizer, model=lm_model, temperature=1.0):
    # full multi-round chats
    chats = batch["prompt"]   # list[list[dict(role, content)]]

    # generate two responses per chat
    responses = extract_dialogue(batch, tokenizer, model, temperature)["generated_response"]

    a1_list, a2_list = [], []
    n = len(responses) // 2

    # render full chats (without add_generation_prompt) for the preference model
    rendered_prompts = [
        tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
        for chat in chats
    ]

    for i in range(n):
        prompt_text = rendered_prompts[i]     # full conversation as plain text
        res1, res2  = responses[2*i], responses[2*i+1]
        a1, a2 = get_preference([prompt_text], [res1], [res2])  # keep your blender API shape
        a1_list.append(a1)
        a2_list.append(a2)

    return {
        "prompt": chats,                                          # <-- keep ALL rounds
        "a1":     [[{"role": "assistant", "content": a[0]}] for a in a1_list],
        "a2":     [[{"role": "assistant", "content": a[0]}] for a in a2_list],
        "rank":   [1] * len(chats),
    }


if __name__ == "__main__":
    from datasets import concatenate_datasets

    repo_id = "august66/drpo_hh_qwen2.5_1.5b"
    chunk_size = 2000
    accum = []

    for start in range(0, len(hh_ds), chunk_size):
        end = min(start + chunk_size, len(hh_ds))
        chunk = hh_ds.select(range(start, end))

        mapped = chunk.map(
            prepare_dataset,
            batched=True,
            batch_size=128,
            remove_columns=chunk.column_names,
            fn_kwargs={"tokenizer": tokenizer, "model": lm_model, "temperature": 1.0},
            desc=f"prepare [{start}:{end})",
        )
        accum.append(mapped)

        # push after each chunk (or every few chunks)
        combined = concatenate_datasets(accum)
        combined.push_to_hub(
            repo_id,
            split="train",
            commit_message=f"Up to {end} examples",
            max_shard_size="500MB",
        )