import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM

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

def strip_prompt(prompt: str, text: str) -> str:
    """
    If `text` literally starts with `prompt` (ignoring leading/trailing
    whitespace), cut that prefix off and return the remainder.
    """
    p = prompt.strip()
    # Escaping safeguards punctuation / regex metacharacters
    pattern = r"^\s*" + re.escape(p) + r"\s*"
    return re.sub(pattern, "", text, count=1).lstrip()



def process_split(original, seed = 42):
    swapped = original.map(lambda x: {
        'a1': x['a2'],
        'a2': x['a1'],
        # 'rank': 1 - int(random.random() < x['chosen_preference']),
        'rank': 1 - x['rank'],
    })

    return concatenate_datasets([original, swapped]).shuffle(seed=seed)


def main():
    seed = 42
    FIRST = 100
    SECOND = 20_000
    data_cache_path = "/workspace/dataset"
    drpo_train = load_dataset("august66/DRPO_data_from_ultrafeed", split="train", cache_dir=data_cache_path)
    drpo_train = process_split(drpo_train)
    drpo_train_reshuffle = drpo_train.shuffle(seed=seed)
    drpo_train_split_1 = drpo_train_reshuffle.select(range(FIRST))
    drpo_train_split_2 = drpo_train_reshuffle.select(range(FIRST, FIRST + SECOND))
    drpo_train_split_3 = drpo_train_reshuffle.select(range(FIRST + SECOND, len(drpo_train_reshuffle)))

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"   # use 0.5B model to test for now 
    cache_path = "/workspace/model_cache"
    model_args = ModelConfig(model_name)
    model_torch_dtype = torch.float16
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

    with open("/workspace/Self_play_DRPO/DRPO_scripts/hh/train_configs/config_gpm.yaml", "r") as f:
        training_args_config = yaml.safe_load(f)


    training_args = DRPOConfig(
        **training_args_config
    )


    training_args.preference_model_id = 'llm-blender/PairRM-hf'

    preference_pipeline = PairRMPipeline(
        model_name_or_path = training_args.preference_model_id,
    )

    trainer = DRPOTrainer(
        model=lm_model_instance,
        ref_model=ref_model,
        preference_model=preference_pipeline,
        train_dataset = drpo_train_split_1,
        processing_class=lm_model_tokenizer,
        args=training_args,
    )

    trainer.train()

if __name__ == "__main__":
    main()
