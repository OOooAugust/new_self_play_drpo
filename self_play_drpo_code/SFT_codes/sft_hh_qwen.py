"""
SFT training script for Qwen2.5-1.5B on HH-Helpful dataset.

Usage:
    # Single GPU
    python sft_hh_qwen.py

    # Multi-GPU (e.g. 2 GPUs)
    nohup accelerate launch --num_processes 1 sft_hh_qwen.py > training.log 2>&1

    # Multi-GPU with config file
    accelerate launch --config_file accelerate_config.yaml sft_hh_qwen.py
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def main():
    # ── Config ──
    model_name = "Qwen/Qwen2.5-1.5B"
    dataset_name = "august66/hh_helpful_base"
    repo_id = "august66/qwen2.5-1.5b-base-hh-helpful-sft"
    model_cache = '/root/autodl-tmp/model_cache'
    data_cache = '/root/autodl-tmp/data_cache'

    # ── 1. Load & prepare dataset ──
    print("Loading dataset...")
    ds_hh_helpful = load_dataset(dataset_name, split="train", cache_dir = data_cache)

    # Expand each row into two conversation rows (prompt+chosen, prompt+rejected)
    all_convos = []
    for ex in ds_hh_helpful:
        all_convos.append({"messages": ex["prompt"] + ex["chosen"]})
        all_convos.append({"messages": ex["prompt"] + ex["rejected"]})

    sft_dataset = Dataset.from_list(all_convos)
    sft_dataset = sft_dataset.shuffle(seed=42)
    print(f"SFT dataset: {len(sft_dataset)} rows (2x {len(ds_hh_helpful)} original)")

    # ── 2. Load model & tokenizer ──
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir = model_cache)
    # Do NOT set device_map="auto" — accelerate handles device placement for DDP
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir = model_cache
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Training config ──
    # SFTTrainer auto-detects the "messages" column and applies
    # tokenizer.apply_chat_template() internally to convert each
    # [{role, content}, ...] conversation into the model's chat format
    # (e.g. <|im_start|>user\n...<|im_end|> for Qwen2.5).
    training_args = SFTConfig(
        output_dir = "/root/autodl-tmp/output/sft_qwen25_hh_helpful",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=2048,
        report_to="none",
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
    )

    # ── 4. Train ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    print(f"Training for {training_args.num_train_epochs} epochs")
    print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size per GPU: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    trainer.train()

    # ── 5. Save ──
    trainer.model.push_to_hub(repo_id)
    trainer.tokenizer.push_to_hub(repo_id)
    print(f"Model saved to {repo_id}")


if __name__ == "__main__":
    main()
