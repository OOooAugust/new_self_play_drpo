import shutil
from trl import apply_chat_template
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers import BitsAndBytesConfig
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    NashMDTrainer,
    NashMDConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if __name__ == "__main__":
    script_args = ScriptArguments(
        dataset_name="Kyleyee/train_data_Helpful_explicit_prompt",
        dataset_train_split="train",
        dataset_test_split ="test",
    )

    training_args = NashMDConfig(
        gradient_checkpointing=False,
        bf16=True,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=8,
        learning_rate=4e-7,
        logging_steps=5,
        max_new_tokens=256,
        max_length=1024,
        temperature=0.66,
        beta=0.04,
        reward_model_path = "Kyleyee/Qwen2.5-1.5B-reward_hh-witheos",
        missing_eos_penalty=1,
        push_to_hub=False,  
        output_dir = "./output/hh/nashmd",
        report_to=["wandb"],
    )
    model_args = ModelConfig(
        model_name_or_path = "Kyleyee/Qwen2.5-1.5B-sft-hh-3e",
    )
    
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

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
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    tokenizer.eos_token = "<|im_end|>"

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
#        ,  **model_kwargs
    )
   
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.remove_columns(["chosen","rejected"])
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split].select(range(500)) if training_args.eval_strategy != "no" else None

    # def prepare_dataset(dataset, tokenizer):
    #     """pre-tokenize the dataset before training; only collate during training"""

    #     def tokenize(element):
    #         input_ids = tokenizer.apply_chat_template(
    #             element["prompt"],
    #             padding=False,
    #             add_generation_prompt=True,
    #         )
    #         return {"input_ids": input_ids, "lengths": len(input_ids)}

    #     return dataset.map(
    #         tokenize,
    #         remove_columns=dataset.column_names,
    #         num_proc=training_args.dataset_num_proc,
    #     )

    # # Compute that only on the main process for faster data processing.
    # # see: https://github.com/huggingface/trl/pull/1255
    # with PartialState().local_main_process_first():
    #     # filtering
    #     train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
    #     if eval_dataset is not None:
    #         eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    # assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    # print("decode example of train_dataset",tokenizer.decode(train_dataset[0]["input_ids"]))
    ################
    # Training
    ################
    trainer = NashMDTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)