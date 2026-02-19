#!/usr/bin/env python
"""
DRPO Training Script for Multi-GPU
Run with: accelerate launch --num_processes=2 drpo_train_multigpu.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import sys
import yaml
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, DatasetDict
from accelerate import PartialState
from accelerate.utils import gather_object

LOCAL_TRL_PARENT = "/root/autodl-tmp/new_self_play_drpo"
if LOCAL_TRL_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_TRL_PARENT)

from trl import DRPOConfig, DRPOTrainerParallel, ModelConfig
from trl.trainer.drpo_utils import BTwithRewardPipeline


def load_model(model_path, task='generation', model_type='decoder', 
               model_cache_path='/root/autodl-tmp/model_cache'):
    model_args = ModelConfig(model_path)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )
    padding_side = 'left' if model_type == 'decoder' else 'right'
    
    if task == 'generation':
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs, cache_dir=model_cache_path)
    else:
        from transformers import AutoModelForSequenceClassification
        model_instance = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, **model_kwargs, cache_dir=model_cache_path)
    
    model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side=padding_side,
        truncation_side=padding_side, use_fast=True, fix_mistral_regex=True,
        trust_remote_code=model_args.trust_remote_code, cache_dir=model_cache_path)
    
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    if getattr(model_instance.config, "pad_token_id", None) is None:
        model_instance.config.pad_token_id = model_tokenizer.pad_token_id
    
    return model_instance, model_tokenizer


def process_split(original, seed=42):
    swapped = original.map(lambda x: {
        'a1': x['a2'], 'a2': x['a1'], 'rank': 1 - x['rank']})
    return concatenate_datasets([original, swapped]).shuffle(seed=seed)



def main():
    # Use PartialState instead of Accelerator to avoid state reset issues
    # The Trainer will create its own Accelerator internally
    state = PartialState()
    local_rank = state.local_process_index
    world_size = state.num_processes
    is_main = state.is_main_process
    
    print(f"[Rank {local_rank}/{world_size}] Using device: {state.device}")
    
    # Paths
    data_cache_path = "/root/autodl-tmp/dataset"
    ds_path = "august66/hh_helpfulness_drpo_from_sft"
    config_path = "/root/autodl-tmp/new_self_play_drpo/self_play_drpo_code/training_config/config_normal_dist_IS_kl.yaml"
    
    # Load training config
    with open(config_path, "r") as f:
        training_args_config = yaml.safe_load(f)
    
    # Override for debugging (remove these lines for full training)
    # training_args_config["logging_steps"] = 1
    # training_args_config["max_steps"] = 10
    
    training_args = DRPOConfig(**training_args_config)
    
    # Load models
    print(f"[Rank {local_rank}] Loading models...")
    ref_policy_model, ref_policy_tokenizer = load_model("august66/qwen2.5-1.5b-base-hh-helpful-sft")
    target_policy_model, _ = load_model("august66/qwen2.5-1.5b-base-hh-helpful-sft")
    dpo_policy_model, _ = load_model('august66/hh_qwen_1.5b_sft_dpo_model')
    print(f"[Rank {local_rank}] Models loaded")
    
    # Load preference pipeline
    print(f"[Rank {local_rank}] Loading preference pipeline...")
    preference_pipeline = BTwithRewardPipeline(
        training_args.preference_model_id, 
        training_args.preference_model_id,
        use_chat_template = True
    )
    print(f"[Rank {local_rank}] Preference pipeline loaded")
    

    ds = load_dataset(ds_path, cache_dir = data_cache_path)

    # If you saved a DatasetDict, load_from_disk returns DatasetDict
    # If you saved a single Dataset, it returns Dataset
    if isinstance(ds, DatasetDict):
        #drpo_train = process_split(concatenate_datasets(list(ds.values())))
        drpo_train = concatenate_datasets(list(ds.values()))
    elif isinstance(ds, Dataset):
        drpo_train = ds
    else:
        raise TypeError(f"Unexpected type from load_from_disk: {type(ds)}")

    drpo_train = drpo_train.shuffle(seed=1234)
    
    if is_main:
        print(f"Starting training with {len(drpo_train)} samples")
        print(f"Logging steps: {training_args.logging_steps}")
        if hasattr(training_args, 'max_steps') and training_args.max_steps:
            print(f"Max steps: {training_args.max_steps}")
    
    # Synchronize before training
    state.wait_for_everyone()
    print(f"[Rank {local_rank}] Creating trainer...")
    
    # Create trainer
    trainer = DRPOTrainerParallel(
        model=target_policy_model,
        ref_model=ref_policy_model,
        dpo_model=dpo_policy_model,
        dpo_as_reward=True,
        preference_model=preference_pipeline,
        train_dataset=drpo_train,
        processing_class=ref_policy_tokenizer,
        args=training_args
    )
    
    # Train
    print(f"[Rank {local_rank}] Starting train()...")
    trainer.train()
    print(f"[Rank {local_rank}] Training complete!")
    
    # Finish wandb on main process
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()
