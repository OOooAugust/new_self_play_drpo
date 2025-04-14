import re
import torch
import os
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, pipeline

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

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    ).to(device)
    return tokenizer, model

def get_checkpoint_paths(base_dir):
    """获取目录下所有checkpoint路径"""
    checkpoints = []
    # 检查基础目录
    if os.path.exists(os.path.join(base_dir, "model.safetensors")):
        print(base_dir)
        checkpoints.append(base_dir)
    
    # 检查checkpoint子目录
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith('checkpoint-')]
    for cp in checkpoint_dirs:
        cp_path = os.path.join(base_dir, cp)
        if os.path.exists(os.path.join(cp_path, "model.safetensors")):
            checkpoints.append(cp_path)
    
    # 按checkpoint号码排序
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if '-' in x else 0)
    return checkpoints

def extract_dialogue(examples: dict, temperature: float, models: dict) -> dict:
    prompts = [" ".join(text.split()[:5]) for text in examples["text"]]
    results = {}
    
    for model_path, (tokenizer, model) in models.items():
        responses = generate_text(prompts, tokenizer, model, temperature)
        
        # 情感分析
        sentiment_results = sentiment_analysis(responses, batch_size=128, truncation=True, padding=True)
        scores = [res["score"] if res["label"] == "POSITIVE" else 1 - res["score"] 
                for res in sentiment_results]
        
        # 使用checkpoint号码作为标识
        name = model_path.split('/')[-1] if 'checkpoint-' in model_path else model_path.split('/')[-1]
        results[f"{name}_responses"] = responses
        results[f"{name}_scores"] = scores
    
    return results

if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset("stanfordnlp/imdb")["test"]
    dataset = dataset.remove_columns(['label'])
    # 使用随机采样的1000个样本
    dataset = dataset.shuffle(seed=44).select(range(1000))
    
    # 指定本地模型目录
    base_dir = "./output/13/"  # 替换为你的模型目录路径
    checkpoint_numbers = ["400", "1000", "2000", "3000", "4000"]
    baseline_models = [
        "Kyleyee/Qwen2-0.5B-DPO-imdb-tm-tp",
        "Kyleyee/Qwen2-0.5B-stf-imdb"
    ]
    model_paths = baseline_models  # base model
    model_paths.extend([os.path.join(base_dir, f"checkpoint-{num}") for num in checkpoint_numbers])

    print(model_paths)
    # 预加载所有模型
    print("Loading models...")
    models = {path: load_model(path) for path in model_paths}
    # print(checkpoint_paths)
    
    # 设置温度参数
    temperatures = [0, 0.25, 0.5, 1.0]
    processed = DatasetDict()
    
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
    # 使用目录名和时间戳作为结果数据集的名称
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    dir_name = os.path.basename(os.path.dirname(base_dir))
    processed.push_to_hub(f"Eehan/eval-{dir_name}-checkpoints-{timestamp}")