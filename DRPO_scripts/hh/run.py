import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import yaml
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers import BitsAndBytesConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from trl.trainer.drpo_utils import GPMPipeline, estDPOStylePipeline, BTRewardNetwork
from trl.trainer import DRPOConfig, DRPOTrainer

def main(script_args, training_args, model_args):
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
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    tokenizer.eos_token = "<|im_end|>"
    print("special tokens", tokenizer.special_tokens_map)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    # value_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    # policy = AutoModelForCausalLM.from_pretrained(
    #     training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    # )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    if training_args.is_bt_model:
        if isinstance(training_args.preference_model_id, dict):
            preference_pipeline = estDPOStylePipeline(training_args.preference_model_id)
        else: 
            preference_pipeline = BTRewardNetwork(training_args.preference_model_id, revision=training_args.preference_model_revision)
    else:
        preference_pipeline = GPMPipeline(training_args.preference_model_id)


    print("ESO TOKEN",tokenizer.eos_token)
    print("ESO TOKEN ID",tokenizer.eos_token_id)
    print(tokenizer.special_tokens_map)

    ################
    # Dataset
    ################
    def transform_dataset(dataset, seed=688):
    # Process each split individually (train/test)
        def process_split(split):
            original = dataset[split]
            swapped = original.map(lambda x: {
                'a1': x['a2'],
                'a2': x['a1'],
                # 'rank': 1 - int(random.random() < x['chosen_preference']),
                'rank': 1 - x['rank'],
            })

            return concatenate_datasets([original, swapped])

    # Apply processing to all splits
        return DatasetDict({
            split: process_split(split).shuffle(seed=seed)
            for split in dataset.keys()  # Handles 'train', 'test', etc.
        })
    dataset = load_dataset(script_args.dataset_name, revision=script_args.dataset_config["revision"])

    ################
    # Training
    ################
    trainer = DRPOTrainer(
        model=model,
        ref_model=ref_model,
        preference_model=preference_pipeline,
        train_dataset = dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    script_args = ScriptArguments(
            dataset_name="Kyleyee/train_data_hh_for_drpo",
            dataset_config={"revision": "6a4692a29dfb1ec436d80763a7f36c9d1c0f33d9"},
            dataset_train_split="train",
            dataset_test_split="validation",
    )

    model_args = ModelConfig(
            model_name_or_path = "Kyleyee/Qwen2.5-1.5B-sft-hh-3e",
    )

    with open("./DRPO_scripts/hh/train_configs/config0.yaml", "r") as f:
        training_args_config = yaml.safe_load(f)

    training_args = DRPOConfig(
        **training_args_config
    )

    main(script_args, training_args, model_args)