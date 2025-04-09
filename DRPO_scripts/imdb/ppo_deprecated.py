
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    is_vision_available,
    GenerationConfig
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
from trl.trainer import PPOConfig, PPOTrainer
from accelerate import PartialState
# from datasets import load_dataset



if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    # script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    # shutil.rmtree(training_args.output_dir, ignore_errors=True)
    model_args = ModelConfig(model_name_or_path = "Kyleyee/Qwen2-0.5B-stf-imdb")
    
    training_args = PPOConfig(
        sft_model_path="Kyleyee/Qwen2-0.5B-stf-imdb",
        num_ppo_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        logging_steps=5,
        save_strategy="no",
        dataset_num_proc=1,
        push_to_hub=False,
        report_to = ['wandb'],
        ds3_gather_for_generation=False,
        output_dir="output/ppo_imdb_kl_only",
    )

    raw_dataset_id = "Kyleyee/train_data_imdb_subsft"

    script_args = ScriptArguments(
            dataset_name=raw_dataset_id,
            dataset_train_split="train",
            dataset_test_split="test",
    )


    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        print("tokenizer.pad_token is None and set to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )

    

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    if len(tokenizer) != policy.config.vocab_size:
        print(policy.config.vocab_size) # 151936
        print("Resizing model embeddings to fit tokenizer vocab size")
        policy.resize_token_embeddings(len(tokenizer))

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
        if len(tokenizer) != ref_policy.config.vocab_size:
            print("Resizing model embeddings to fit tokenizer vocab size")
            ref_policy.resize_token_embeddings(len(tokenizer))
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"



    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""
        def transform_dataset(dataset):
            dataset = dataset.remove_columns(["label", "text", "chosen", "rejected", "a_1", "a_2", "a_1_preference", "a_2_preference", "rejected_preference", "chosen_preference"])
            return dataset

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}
        
        dataset = transform_dataset(dataset)

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )
    
    


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        print(train_dataset[0:2])
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()






##################################

##################################
