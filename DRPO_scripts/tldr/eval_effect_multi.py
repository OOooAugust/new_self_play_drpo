import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, DatasetDict
from huggingface_hub import ModelCard
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, pipeline


def pipe(model_id):
    processing_class = AutoTokenizer.from_pretrained(model_id)
    processing_class.padding_side = "left"
    processing_class.add_special_tokens({"pad_token": "[PAD]"})
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=processing_class,
        batch_size=64,
        eos_token_id=processing_class.eos_token_id,
    )
    return pipe, processing_class

def extract_dialogue(examples: dict, temperature: float, kwargs:dict) -> dict:
    prompts = examples["prompt"]
    prompt_list = []
    for text in prompts:
        pattern = r"POST:\s*(.*?)(?=\s*\nTL;DR:|$)"
        match = re.search(pattern, text, re.DOTALL)
        prompt_list.append(match.group(1).strip())
    
    for model_name, model_kwargs in kwargs.items():
        kwargs
        kwargs[model_name]["do_sample"] = temperature > 0
        if temperature > 0:
            model_kwargs["temperature"] = temperature
        generated = pipe(model_name, **model_kwargs)(prompts, **model_kwargs)
        examples[model_name] = [batch[0]["generated_text"] for batch in generated]

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
    # 加载数据集
    dataset = load_dataset("Kyleyee/train_data_tldr_for_drpo")["validation"]


    dataset = dataset.remove_columns(["rank", "a1", "a2"])
    dataset = deduplicate_consecutive_dataset(dataset)
    dataset = dataset.select(range(1000))

    temperatures = [0, 0.3, 0.7]

    dpo_pipe, dpo_tokenizer = pipe("Kyleyee/pythia-1b-deduped-tldr-dpo")
    drpo_pipe, drpo_tokenizer = pipe("Eehan/pythia-1b-deduped-tldr-drpo-0.9tmp")
    sft_pipe, sft_tokenizer = pipe("trl-lib/pythia-1b-deduped-tldr-sft")

    kwargs = {
        "dpo": {
            "max_new_tokens": 64,
            "eos_token_id": dpo_tokenizer.eos_token_id,
            "return_full_text": False,
        },
        "drpo-0.9tmp": {
            "max_new_tokens": 64,
            "eos_token_id": drpo_tokenizer.eos_token_id,
            "return_full_text": False,
        },
        "sft": {
            "max_new_tokens": 64,
            "eos_token_id": sft_tokenizer.eos_token_id,
            "return_full_text": False,
        },
    }

    # 创建包含不同温度子集的DatasetDict
    processed = DatasetDict()
    for idx, temp in enumerate(temperatures):
        processed_shard = dataset.map(
            extract_dialogue,
            fn_kwargs={"temperature": temp, "kwargs": kwargs},
            batched=True,
            batch_size=64
        )
        processed[f"temperature_{temp}"] = processed_shard

    processed.push_to_hub("Eehan/eval-tldr-dpo-drpo-0.9tmp-sft")