"""
Reward model training: Qwen2.5-1.5B on HH-Helpful dataset.

Usage:
    python reward_hh_qwen.py

Multi-GPU:
    accelerate launch reward_hh_qwen.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig


def main():
    # ── Config ──
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name = "august66/hh_helpful_base"
    output_dir = "./reward_qwen25_hh_helpful"
    data_cache = "/Users/august/Documents/LLM Research/new_self_play_drpo/data_cache"

    # ── 1. Load dataset ──
    print("Loading dataset...")
    ds = load_dataset(dataset_name, cache_dir=data_cache, split="train")

    # ── 2. Format for RewardTrainer: chosen/rejected as full conversations ──
    def format_for_reward(example):
        return {
            "chosen": example["prompt"] + example["chosen"],
            "rejected": example["prompt"] + example["rejected"],
        }

    reward_dataset = ds.map(format_for_reward, remove_columns=["prompt"])
    reward_dataset = reward_dataset.shuffle(seed=42)

    # Split off a small eval set (5%)
    split = reward_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} pairs, Eval: {len(eval_dataset)} pairs")

    # ── 3. Load model as reward model (sequence classification with 1 label) ──
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── 4. Training config ──
    # RewardTrainer auto-applies tokenizer.apply_chat_template() on the
    # chosen/rejected conversation lists, then trains the model to assign
    # higher scalar reward to chosen vs rejected.
    training_args = RewardConfig(
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
        max_length=1024,
        eval_strategy="epoch",
        report_to="none",
    )

    # ── 5. Train ──
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"Training reward model for {training_args.num_train_epochs} epochs on {len(train_dataset)} pairs, eval on {len(eval_dataset)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    trainer.train()

    # ── 6. Save ──
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Reward model saved to {output_dir}")


if __name__ == "__main__":
    main()
