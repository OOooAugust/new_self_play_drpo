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

def extract_dialogue(examples: dict, temperature: float, pipes: dict, kwargs:dict) -> dict:
    prompts = examples["prompt"]
    prompt_list = []
    for text in prompts:
        pattern = r"POST:\s*(.*?)(?=\s*\nTL;DR:|$)"
        match = re.search(pattern, text, re.DOTALL)
        prompt_list.append(match.group(1).strip())
    
    for model_name, model_kwargs in kwargs.items():
        kwargs[model_name]["do_sample"] = temperature > 0
        if temperature > 0:
            model_kwargs["temperature"] = temperature
        generated = pipes[model_name](prompts, **model_kwargs)
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
    # Load dataset
    dataset = load_dataset("Kyleyee/train_data_tldr_for_drpo")["validation"]


    dataset = dataset.remove_columns(["rank", "a1", "a2"])
    dataset = deduplicate_consecutive_dataset(dataset)
    dataset = dataset.select(range(1000))

    temperatures = [0, 0.25, 0.5,0.75,1.0]

    # Initialize models and tokenizers
    dpo_pipe, dpo_tokenizer = pipe("Kyleyee/pythia-1b-deduped-tldr-dpo")
    ppo_pipe, ppo_tokenizer = pipe("cleanrl/EleutherAI_pythia-1b-deduped__ppo__tldr")
    # drpo_lowbeta_pipe, drpo_lowbeta_tokenizer = pipe("Eehan/Pythia-1b-deduped-tldr-drpo-temp0.75")
    drpo_lowtemp_pipe, drpo_lowtemp_tokenizer = pipe("Eehan/Pythia-1b-deduped-tldr-drpo-temp-0.25-beta-0.05")
    drpo_hightemp_pipe, drpo_hightemp_tokenizer = pipe("Eehan/Pythia-1b-deduped-tldr-drpo-temp-0.75-beta-0.05")
    dm_pipe, dm_tokenizer = pipe("Eehan/Pythia-1b-deduped-tldr-dm-temp-0.75-beta-0.05")
    sft_pipe, sft_tokenizer = pipe("cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")

    model_names = [
        "dpo",
        "ppo",
        "drpo_lowtemp",
        "drpo_hightemp",
        "dm",
        "sft",
    ]

    # Create dictionaries for pipes and tokenizers
    pipes = {
        "dpo": dpo_pipe,
        "ppo": ppo_pipe,
        "drpo_lowtemp": drpo_lowtemp_pipe,
        "drpo_hightemp": drpo_hightemp_pipe,
        "dm": dm_pipe,
        "sft": sft_pipe,
    }

    tokenizers = {
        "dpo": dpo_tokenizer,
        "ppo": ppo_tokenizer,
        "drpo_lowtemp": drpo_lowtemp_tokenizer,
        "drpo_hightemp": drpo_hightemp_tokenizer,
        "dm": dm_tokenizer,
        "sft": sft_tokenizer,
    }
    
    # Create kwargs dictionary
    kwargs = {
        model_name: {
            "max_new_tokens": 64,
            "eos_token_id": tokenizers[model_name].eos_token_id,
            "return_full_text": False
        } for model_name in model_names
    }

    # Process dataset
    processed = DatasetDict()
    for idx, temp in enumerate(temperatures):
        processed_shard = dataset.map(
            extract_dialogue,
            fn_kwargs={"temperature": temp, "pipes": pipes, "kwargs": kwargs},
            batched=True,
            batch_size=64
        )
        processed[f"temperature_{temp}"] = processed_shard

    processed.push_to_hub("Eehan/eval-tldr-dpo-ppo-drpo-dm-sft-3000")
