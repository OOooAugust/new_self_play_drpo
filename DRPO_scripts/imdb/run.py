import os

import sys
import os

import yaml

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import random
import torch
from datasets import Dataset, features, load_dataset, concatenate_datasets, DatasetDict
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
        print("Adding pad token")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    else:
        print("Pad token already exists")

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

    def transform_dataset(dataset):
    # Process each split individually (train/test)
        def process_split(split):
            split_dataset = dataset[split]
            split_dataset = split_dataset.rename_column("chosen", "a1")
            split_dataset = split_dataset.rename_column("rejected", "a2")
            split_dataset = split_dataset.remove_columns(["label", "text", "a_1", "a_2", "a_1_preference", "a_2_preference"])

            # Original examples
            original = split_dataset.map(lambda x: {**x, 'rank': int(random.random() < x['chosen_preference'])})
            original = original.remove_columns(["chosen_preference","rejected_preference"])

            # Swapped examples
            swapped = original.map(lambda x: {
                'a1': x['a2'],
                'a2': x['a1'],
                # 'rank': 1 - int(random.random() < x['chosen_preference']),
                'rank': 1 - x['rank'],
            })

            return concatenate_datasets([original, swapped])

        # Apply processing to all splits
        return DatasetDict({
            split: process_split(split).shuffle(seed=688)
            for split in dataset.keys()  # Handles 'train', 'test', etc.
        })

    def extract_dialogue(examples: dict):
        prompt = [" ".join(text.split()[:2]) for text in examples["prompt"]]
        a1 = [" " + " ".join(text.split()[2:]) for text in examples["a1"]]
        a2 = [" " + " ".join(text.split()[2:]) for text in examples["a2"]]
        return {
            "prompt": prompt,
            "a1": a1,
            "a2": a2,
        }

    raw_dataset = load_dataset(script_args.dataset_name, "default")
    dataset = transform_dataset(raw_dataset)
    dataset = dataset.map(extract_dialogue, batch_size=128,batched=True)

    print(f"\033[32mLoaded and transformed dataset sample:\033[0m {dataset['train'][0]}")

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



##################################

script_args = ScriptArguments(
        dataset_name="Kyleyee/train_data_imdb_for_drdpo_preference",
        dataset_train_split="train",
        dataset_test_split="test",
)

model_args = ModelConfig(
        model_name_or_path = "Kyleyee/Qwen2-0.5B-stf-imdb"
)

with open("./DRPO_scripts/imdb/train_configs/config_noisyGT_refT_2.yaml", "r") as f:
    training_args_config = yaml.safe_load(f)

training_args = DRPOConfig(
    **training_args_config
)


if __name__ == "__main__":
    main(script_args, training_args, model_args)
    # shutdown the system
    # os.system("usr/bin/shutdown")



