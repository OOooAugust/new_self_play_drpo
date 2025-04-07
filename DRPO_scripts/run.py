
import os

import sys
import os

import yaml

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
import torch
from datasets import Dataset, features, load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    is_vision_available,
)
from transformers.testing_utils import require_peft, require_torch_gpu_if_bnb_not_multi_backend_enabled, require_vision

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from trl.trainer.drpo_utils import GPMPipeline, BTPipeline, estDPOStylePipeline
from trl.trainer import DRPOConfig, DRPOTrainer

def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    if training_args.is_bt_model:
        if isinstance(training_args.preference_model_id, dict):
            preference_pipeline = estDPOStylePipeline(training_args.preference_model_id)
        else: 
            preference_pipeline = BTPipeline(training_args.preference_model_id)
    else:
        preference_pipeline = GPMPipeline(training_args.preference_model_id)

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # dataset = dataset.remove_columns(["label","text"])
    # print(f"Loaded dataset sample: {dataset['train'][0]}")

    ##########
    # Training
    ################
    trainer = DRPOTrainer(
        model = model,
        ref_model = ref_model,
        preference_model = preference_pipeline,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split].select(range(1000)) if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        args = training_args
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)



def transform_dataset(dataset):
    dataset = dataset.rename_column("chosen", "a1")
    dataset = dataset.rename_column("rejected", "a2")
    dataset = dataset.remove_columns(["label", "text", "a_1", "a_2", "a_1_preference", "a_2_preference", "rejected_preference"])
    dataset = dataset.map(lambda x: {**x, 'rank': int(random.random() < x['chosen_preference'])})
    dataset = dataset.remove_columns(["chosen_preference"])
    return dataset

##################################

##################################

model_id = "Kyleyee/Qwen2-0.5B-stf-imdb"
raw_dataset_id = "Kyleyee/train_data_imdb_subsft"


raw_dataset = load_dataset(raw_dataset_id, "default")
dataset = transform_dataset(raw_dataset)
print(f"Loaded dataset sample: {dataset['train'][0]}")


script_args = ScriptArguments(
        dataset_name=raw_dataset_id,
        dataset_train_split="train",
        dataset_test_split="test",
)

model_args = ModelConfig(
        model_name_or_path = model_id,
)

with open("./DRPO_scripts/train_configs/config10.yaml", "r") as f:
    training_args_config = yaml.safe_load(f)

training_args = DRPOConfig(
    **training_args_config
)


if __name__ == "__main__":
    main(script_args, training_args, model_args)
    # shutdown the system
    # os.system("usr/bin/shutdown")



