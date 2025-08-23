import os
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
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

from trl.trainer.drpo_utils import GPMwithRewardNetwork, estDPOStylePipeline, BTRewardNetwork, PairRMPipeline
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

def process_split(original, seed = 42):
    swapped = original.map(lambda x: {
        'a1': x['a2'],
        'a2': x['a1'],
        # 'rank': 1 - int(random.random() < x['chosen_preference']),
        'rank': 1 - x['rank'],
    })

    return concatenate_datasets([original, swapped]).shuffle(seed=seed)

seed = 1234
data_cache_path = "/workspace/dataset"
model_cache_path = '/workspace/model_cache'
drpo_train = load_dataset("august66/drpo_ultrafeedback_qwen2.5-1.5b_first_iter_20k", split="train", cache_dir=data_cache_path)
drpo_train = process_split(drpo_train)
drpo_train = drpo_train.shuffle(seed=seed)


device = 'cuda'
model_name = "Qwen/Qwen2.5-1.5B-Instruct"   # use 0.5B model to test for now 
cache_path = "/workspace/model_cache"
model_args = ModelConfig(model_name)
model_torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
model_args.trust_remote_code = True
model_kwargs = dict(
    revision = model_args.model_revision,
    torch_dtype = model_torch_dtype, 
    trust_remote_code = model_args.trust_remote_code,
)
lm_model_instance = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
    cache_dir = cache_path,
)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
    cache_dir = cache_path,
)

lm_model_tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, 
    padding_side = 'left', 
    use_fast = True,
    trust_remote_code = model_args.trust_remote_code,
    cache_dir = cache_path
)


if not lm_model_tokenizer.pad_token:
    lm_model_tokenizer.pad_token = lm_model_tokenizer.eos_token

if lm_model_tokenizer.chat_template is None:
    lm_model_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE



with open("/workspace/Self_play_DRPO/DRPO_scripts/hh/train_configs/config_gpm.yaml", "r") as f:
    training_args_config = yaml.safe_load(f)


training_args = DRPOConfig(
    **training_args_config
)


preference_pipeline = PairRMPipeline(
    model_name_or_path = training_args.preference_model_id,
)


trainer = DRPOTrainer(
    model=lm_model_instance,
    ref_model=ref_model,
    preference_model=preference_pipeline,
    train_dataset = drpo_train,
    processing_class=lm_model_tokenizer,
    args=training_args
)

if __name__ == "__main__":
    trainer.train()
