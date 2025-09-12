import os
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,DebertaV2ForSequenceClassification, GPTNeoXForCausalLM
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import DataCollatorWithPadding
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch.nn.functional as F
from tqdm import tqdm
import re
import yaml

LOCAL_TRL_PARENT = "/workspace/Self_play_DRPO"
if LOCAL_TRL_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_TRL_PARENT)

    
# now the import will use your local copy:
from trl import (
    DPOTrainer,
    DPOConfig,
    ModelConfig,
    DRPOTrainer,
    DRPOConfig,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.data_utils import apply_chat_template
from trl.trainer.drpo_utils import PairRMPipeline

def process_split(original, seed = 42):
    swapped = original.map(lambda x: {
        'a1': x['a2'],
        'a2': x['a1'],
        # 'rank': 1 - int(random.random() < x['chosen_preference']),
        'rank': 1 - x['rank'],
    })

    return concatenate_datasets([original, swapped]).shuffle(seed=seed)

def load_model(model_path, task = 'generation', model_type = 'decoder', model_cache_path =  '/workspace/model_cache'):

    model_args = ModelConfig(model_path)
    model_torch_dtype = torch.bfloat16
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



data_cache_path = "/workspace/dataset"
model_cache_path = '/workspace/model_cache'
ds_path = 'august66/drpo_hh_qwen2.5_1.5b'
ref_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 
target_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 
dpo_policy_path = 'august66/hh_qwen_1.5b_dpo_model_2'

#load training argument for drpo
with open("/workspace/Self_play_DRPO/self_play_drpo_code/training_config/config_normal_dist.yaml", "r") as f:
    training_args_config = yaml.safe_load(f)


if __name__ == '__main__':
    seed = 1234
    ref_policy_model, ref_policy_tokenizer = load_model(ref_policy_path)
    target_policy_model, target_policy_tokenizer = load_model(target_policy_path)
    dpo_policy_model, dpo_policy_tokenizer = load_model(dpo_policy_path)
    drpo_train = load_dataset(ds_path, cache_dir=data_cache_path, split = 'train')
    drpo_train = process_split(drpo_train)
    drpo_train = drpo_train.shuffle(seed=seed)

    training_args = DRPOConfig(
        **training_args_config
    )
    preference_pipeline = PairRMPipeline(
        model_name_or_path = training_args.preference_model_id,
    )

    trainer = DRPOTrainer(
        model=target_policy_model,
        ref_model=ref_policy_model,
        dpo_model = dpo_policy_model,
        preference_model=preference_pipeline,
        train_dataset = drpo_train,
        processing_class=ref_policy_tokenizer,
        args=training_args
    )
    trainer.train()