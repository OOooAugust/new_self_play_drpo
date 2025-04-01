
import re
import torch
# from dataclasses import dataclass
# from typing import Optional
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, pipeline
# from trl.data_utils import apply_chat_template

# 初始化设备和模型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 情感分析模型
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

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
    
    outputs = model.generate(**inputs, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def extract_dialogue(examples: dict, temperature: float, models: dict) -> dict:
    prompts = [" ".join(text.split()[:5]) for text in examples["text"]]
    results = {}
    
    for model_name, (tokenizer, model) in models.items():
        responses = generate_text(prompts, tokenizer, model, temperature)
        
        # 情感分析
        sentiment_results = sentiment_analysis(responses, batch_size=128, truncation=True, padding=True)
        scores = [res["score"] if res["label"] == "POSITIVE" else 1 - res["score"] 
                for res in sentiment_results]
        
        # 清理模型名称作为字段名
        safe_name = model_name.replace("Kyleyee/Qwen2-0.5B-", "")
        # safe_name = model_name.replace("/", "_")
        # results[f"{safe_name}_responses"] = responses
        results[f"{safe_name}_scores"] = scores
    
    return results

if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")["test"]
    dataset = dataset.remove_columns(['label'])
    # use randomly sampled 1000 samples
    dataset = dataset.shuffle(seed=42).select(range(5000))
    
    # 初始化模型列表
    model_names = [
        "Kyleyee/Qwen2-0.5B-DPO-imdb-tm-tp",
        "Eehan/Qwen2-0.5B-drpo-imdb-default-1",
        "Eehan/Qwen2-0.5B-drpo-imdb-default-3",
        "Eehan/Qwen2-0.5B-drpo-imdb-indifferent-4",
        "Eehan/Qwen2-0.5B-drpo-imdb-loss1_only-5",
        "Eehan/Qwen2-0.5B-drpo-imdb-loss2_only-6",
        "Eehan/Qwen2-0.5B-drpo-imdb-est_dpo_style-7"
    ]
    
    # 预加载所有模型
    print("Loading models...")
    models = {name: load_model(name) for name in model_names}
    
    # 设置温度参数
    temperatures = [0, 0.25, 0.5, 0.75, 1]
    processed = DatasetDict()
    
    # 遍历每个温度
    for temp in temperatures:
        print(f"Processing temperature {temp}...")
        processed_shard = dataset.map(
            extract_dialogue,
            fn_kwargs={
                "temperature": temp,
                "models": models
            },
            batched=True,
            batch_size=64,
            num_proc=1  
        )
        processed[f"temperature_{temp}"] = processed_shard
    
    # 推送结果到Hub
    print("Pushing to Hub...")
    processed.push_to_hub("Eehan/eval-imdb-drpo-134567-dpo-5000")
    