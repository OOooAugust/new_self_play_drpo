"""
SFT training script for Qwen2.5-1.5B on HH-Helpful dataset.

Usage:
    python sft_hh_qwen.py

Multi-GPU:
    accelerate launch sft_hh_qwen.py
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def main():
    # ── Config ──
    model_name = "Qwen/Qwen2.5-1.5B"
    dataset_name = "august66/hh_helpful_base"
    output_dir = "./sft_qwen25_hh_helpful"
    data_cache = "/Users/august/Documents/LLM Research/new_self_play_drpo/data_cache"

    # ── 1. Load & prepare dataset ──
    print("Loading dataset...")
    ds_hh_helpful = load_dataset(dataset_name, cache_dir=data_cache, split="train")

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Training config ──
    # SFTTrainer auto-detects the "messages" column and applies
    # tokenizer.apply_chat_template() internally to convert each
    # [{role, content}, ...] conversation into the model's chat format
    # (e.g. <|im_start|>user\n...<|im_end|> for Qwen2.5).
    training_args = SFTConfig(
        output_dir=output_dir,
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
        max_seq_length=1024,
        report_to="none",
    )

    # ── 4. Train ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    print(f"Training for {training_args.num_train_epochs} epochs")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    trainer.train()

    # ── 5. Save ──
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
