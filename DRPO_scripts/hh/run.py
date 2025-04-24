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
    ) # FIXME: why do we need padding_side="left"

    tokenizer.eos_token = "<|im_end|>"

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
            preference_pipeline = BTRewardNetwork(training_args.preference_model_id)
    else:
        preference_pipeline = GPMPipeline(training_args.preference_model_id)


    print("ESO TOKEN",tokenizer.eos_token)
    print("ESO TOKEN ID",tokenizer.eos_token_id)
    print(tokenizer.special_tokens_map)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
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
    #     train_dataset = prepare_dataset(train_dataset, tokenizer)
    #     if eval_dataset is not None:
    #         eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    #     # filtering
    #     train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
    #     if eval_dataset is not None:
    #         eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    # assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    # print("decode example of train_dataset",tokenizer.decode(train_dataset[0]["input_ids"]))
    # FIXME: why they are not different
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

    trainer.generate_completions() # FIXME: why do we need this?