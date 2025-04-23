import sys
import os

# # Add the parent directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from trl import apply_chat_template
from transformers import AutoTokenizer
import argparse
from trl.trainer.utils import SIMPLE_SFT_CHAT_TEMPLATE
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from numpy import random
import numpy as np
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)




def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        print('pad_token is None')
        tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE
  
    print("ESO TOKEN",tokenizer.eos_token)
    print("ESO TOKEN ID",tokenizer.eos_token_id)
    print(tokenizer.special_tokens_map)



    ################
    # Dataset
    ################
    dataset = load_dataset("Kyleyee/train_data_HH_sft_CompletionOnly")
    def formatting_prompts_func(example):
        text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"

        return text

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    def inspect_one(raw_text):
        # token 化
        enc = tokenizer(raw_text, add_special_tokens=False)['input_ids']
        # 送进 collator ——> 得到带 -100 mask 的 labels
        batch = collator([enc])
        # 把被监督的 token（label ≠ -100）挑出来
        lbl_ids = [tid for tid, lbl in zip(batch["input_ids"][0], batch["labels"][0]) if lbl != -100]
        print("===== LABEL SEGMENT START =====")
        print(tokenizer.decode(lbl_ids)[:400])      # 只截前 400 字符，够检查了
        print("===== LABEL SEGMENT END =====")

    sample = formatting_prompts_func(dataset["train"][0]) 
    inspect_one(sample)


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

script_args = ScriptArguments(
    dataset_name="Kyleyee/train_data_SFT_Helpful",  
    dataset_train_split="train",
    dataset_test_split="test",
) 

training_args = SFTConfig(
    output_dir="./selfmodel/output/Qwen2.5-1.5B-instruct-sft-hh-3e-CompletionOnly-witheos",
    bf16=True,
    num_train_epochs=3,
    learning_rate=2.0e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=True,  
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=1000,
    hub_model_id="Kyleyee/Qwen2.5-1.5B-instruct-sft-hh-3e-CompletionOnly-witheos",  
    report_to=["wandb"]
)

model_args = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct", 
    torch_dtype="auto",
    trust_remote_code=True,
)


if __name__ == "__main__":
    main(script_args, training_args, model_args)