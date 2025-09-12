import os
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch.nn.functional as F
from tqdm import tqdm
import re
import yaml
from datasets import load_dataset

LOCAL_TRL_PARENT = "/workspace/Self_play_DRPO"
if LOCAL_TRL_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_TRL_PARENT)

    
# now the import will use your local copy:
from trl import (
    DPOTrainer,
    DPOConfig,
    ModelConfig
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.data_utils import apply_chat_template as trl_apply

data_cache_path = "/workspace/dataset"
model_cache_path = '/workspace/model_cache'
ds_path = 'august66/drpo_hh_qwen2.5_1.5b'
ref_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 
target_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 


def load_model(model_path, task = 'generation', model_type = 'decoder', model_cache_path =  '/workspace/model_cache'):

    model_args = ModelConfig(model_path)
    model_torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
    model_kwargs = dict(
    revision = model_args.model_revision,
    torch_dtype = model_torch_dtype, 
    trust_remote_code = model_args.trust_remote_code,
    )

    padding_side = 'left' if model_type == 'decoder' else 'right'
    truncation_side = 'left' if model_type == 'decoder' else 'right'

    if task == 'generation':
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
            cache_dir = model_cache_path,
        )

    elif task == 'reward':
        model_instance = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
            cache_dir = model_cache_path,
        )
    

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side = padding_side, 
        truncation_side = truncation_side,
        use_fast = True,
        trust_remote_code = model_args.trust_remote_code,
        cache_dir = model_cache_path
    )

    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    if getattr(model_instance.config, "pad_token_id", None) is None:
        model_instance.config.pad_token_id = model_tokenizer.pad_token_id

    if model_tokenizer.eos_token is None:
        model_tokenizer.eos_token = model_tokenizer.pad_token  

    if getattr(model_instance.config, "eos_token_id", None) is None:
        model_instance.config.eos_token_id = model_tokenizer.eos_token_id

    return model_instance, model_tokenizer


def render_prompt(history):
    # Prefer tokenizer's own method if available
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    return trl_apply(history, tok, add_generation_prompt=True)

def as_assistant_text(x):
    # your a1/a2 can be list[{"role":"assistant","content":...}] or plain string
    return x[0]["content"] if isinstance(x, list) else x

def to_three_cols(ex):
    prompt_text = render_prompt(ex["prompt"])
    a1 = as_assistant_text(ex["a1"])
    a2 = as_assistant_text(ex["a2"])
    if ex["rank"] == 1:   # adjust if your convention differs
        chosen, rejected = a1, a2
    else:
        chosen, rejected = a2, a1
    return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}



if __name__ == "__main__":

    seed = 1234
    ref_policy_model, ref_policy_tokenizer = load_model(ref_policy_path)
    target_policy_model, target_policy_tokenizer = load_model(target_policy_path)
    tok = target_policy_tokenizer
    drpo_train = load_dataset("august66/drpo_hh_qwen2.5_1.5b", split="train")
    drpo_train = drpo_train.map(to_three_cols, remove_columns=[c for c in drpo_train.column_names
                                           if c not in {"prompt","chosen","rejected"}])
    print ({
        'prompt': drpo_train[0]['prompt'],
        'chosen': drpo_train[0]['chosen'],
        'rejected': drpo_train[0]['rejected']
    })

    config = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        report_to="wandb",          
        run_name="my-dpo-run-hh-qwen-1.5b",
        logging_steps=20,
        max_prompt_length = 512,
        max_completion_length = 512,
        max_length = 1024,
        output_dir = 'dpo_out',
        save_strategy = 'steps',
        save_steps = 10000,
        save_total_limit = 1,
        push_to_hub = True,
        hub_model_id = 'august66/hh_qwen_1.5b_dpo_model_2',
        hub_strategy = 'every_save',
        bf16 = True,
        fp16 = False
    )
    trainer = DPOTrainer(
        model=target_policy_model,
        ref_model=ref_policy_model,     
        args=config,
        train_dataset=drpo_train,
        processing_class=ref_policy_tokenizer
    )
    trainer.train()