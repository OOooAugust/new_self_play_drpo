
import gc
import math
import os
import re
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union, Any, List, Dict, Callable, EvalPrediction
import warnings

import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCasualLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available, is_torch_xpu_available
from transformers.data.data_collator import DataCollatorMixin

from datasets import DatasetDict, load_dataset
from dataclasses import dataclass, field

from safetensors.torch import load_file
from huggingface_hub import snapshot_download, ModelCard

from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .drpo_config import DRPOConfig
from .utils import (
    OnlineTrainerState,
    pad,
    pad_to_length,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
    flush_left
)
from dataclasses import dataclass


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb




INVALID_LOGPROB = 1.0
@dataclass
class DataCollatorForDRPO(DataCollatorMixin):
    '''
    Data collator for double robust preference learning.

    tokenized examples -> padded batch

    Examples:
    ```python
    >>> examples = [
        {"prompt_input_ids": [1, 2, 3], 
        "a1_input_ids": [4, 5, 6], 
        "a2_input_ids": [7, 8, 9], 
        
        "rank": 1.,},

        {"prompt_input_ids": [1, 2],
        "a1_input_ids": [4, 5],
        "a2_input_ids": [7, 8],

        "rank": 0.,}
    ]

    >>> collator(examples)
    {
        "prompt_input_ids": tensor([[1, 2, 3], [1, 2, 0]]),
        "prompt_attention_mask": tensor([[1, 1, 1], [1, 1, 0]]),
        "a1_input_ids": tensor([[4, 5, 6], [4, 5, 0]]),
        "a1_attention_mask": tensor([[1, 1, 1], [1, 1, 0]]),
        "a2_input_ids": tensor([[7, 8, 9], [7, 8, 0]]),
        "a2_attention_mask": tensor([[1, 1, 1], [1, 1, 0]]),
        
        "rank": tensor([1., 0.]),
    }
    ```
    '''
    pad_token_id: int
    return_tensors: str = "pt"
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        a1_input_ids = [torch.tensor(example["a1_input_ids"]) for example in examples]
        a1_attention_mask = [torch.ones_like(input_ids) for input_ids in a1_input_ids]
        a2_input_ids = [torch.tensor(example["a2_input_ids"]) for example in examples]
        a2_attention_mask = [torch.ones_like(input_ids) for input_ids in a2_input_ids]
        # if "pixel_values" in examples[0]:
        #     pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        # if "pixel_attention_mask" in examples[0]:
        #     pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        
        # FIXME: check if the ref_a1_logps is in corresponding format as a float16 only.
        if "ref_a1_logps" in examples[0]: # and "ref_a2_logps" in examples[0]
            ref_a1_logps = torch.tensor([example["ref_a1_logps"] for example in examples])
            # ref_a2_logps = torch.tensor([example["ref_a2_logps"] for example in examples])
            output["ref_a1_logps"] = ref_a1_logps

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["a1_input_ids"] = pad(a1_input_ids, padding_value=self.pad_token_id)
        output["a1_attention_mask"] = pad(a1_attention_mask, padding_value=0)
        output["a2_input_ids"] = pad(a2_input_ids, padding_value=self.pad_token_id)
        output["a2_attention_mask"] = pad(a2_attention_mask, padding_value=0)

        # if "pixel_values" in examples[0]:
        #     output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        # if "pixel_attention_mask" in examples[0]:
        #     output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        # if "image_sizes" in examples[0]:
        #     output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "a1_preference" in examples[0]:
            output["a1_preference"] = torch.tensor([example["a1_preference"] for example in examples]) 
        if "rank" in examples[0]:
            output["rank"] = torch.tensor([example["rank"] for example in examples])
        
        
        # output["ref_a2_logps"] = ref_a2_logps


        return output


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
# class PolicyAndValueWrapper(nn.Module):
#     def __init__(self, policy, value_model) -> None:
#         super().__init__()
#         self.policy = policy
#         self.value_model = value_model
#         self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

#     def forward(self, **kwargs):
#         output = self.critic_backbone(**kwargs)
#         logits = self.value_model.score(output.hidden_states[-1])
#         return self.policy(**kwargs), logits

class DRPOTrainer(Trainer):
    r"""
    Initialize a DRPOTrainer.

    Args:
        args (:class:`~transformers.DRPOConfig`):
            The configuration object used to control the training.
        model (:class:`~transformers.PreTrainedModel`):
            The model to train, evaluate, and generate from.
        ref_model (:class:`~transformers.PreTrainedModel`, `optional`):
            The reference model to compute the reference log probabilities. If `None`, the model is used as the reference
            model.
        preference_model (:class:`~transformers.PreTrainedModel`):
            The preference model to compute the preference probabilities.
        train_dataset (:class:`~datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (:class:`~datasets.Dataset`, `optional`):
            The dataset to use for evaluation.
        processing_class (:class:`~transformers.PreTrainedTokenizerBase`, `optional`):
            The processing class used to process the data.
        data_collator (:class:`~transformers.DataCollator`, `optional`):
            The data collator to use for training.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model. 
    """
    _tag_names = ["trl", "drpo"]
    def __init__(
            self,
            args: DRPOConfig,
            model: nn.Module,
            ref_model: Optional[nn.Module],
            preference_model: nn.Module,
            train_dataset: Dataset,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None,
            data_collator: Optional[Union[DataCollatorWithPadding, DataCollatorMixin,DataCollatorForDRPO]] = None,
            model_init: Optional[Callable[[], nn.Module]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        ) -> None:
            ########################
            # some preparing steps
            ########################
        if model is None:
            raise ValueError("`model` must be provided. ")
        
        if not isinstance(model, str) and ref_model is model:
            raise ValueError("`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the same as `model`, you must make a copy of it, or `None` if you use peft.")
        
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DRPOTrainer/DRPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DRPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
            "You passed ref_model_init_kwargs to the DRPOTrainer/DRPOConfig, but your ref_model is already instantiated."
        )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DRPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            model = AutoModelForCasualLM.from_pretrained(model, **model_init_kwargs)
        if isinstance(ref_model, str):
            ref_model = AutoModelForCasualLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if args.generate_during_eval and not (is_wandb_available()):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
                " Please install `wandb` or `comet-ml` to resolve."
            )
        
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        self.mdoel_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        elif args.precomnpute_reference:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # TODO: check what preference model is like and do some checks
        self.preference_model = preference_model

        if processing_class is None:
            raise ValueError("`processing_class` must be provided to tokenize the dataset.")
        self.processing_class = processing_class

        if args.padding_value is not None:
            self.padding_value = args.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError("padding value is neither specified in DRPOConfig, Nor in the processing class.")
        
        if data_collator is None:
            data_collator = DataCollatorForDRPO(pad_token_id=self.padding_value)

        if args.generation_config is not None:
            self.generation_config = GenerationConfig.from_dict(args.generation_config)

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_logits_to_keep = args.use_logits_to_keep

        # self.generate_during_eval = args.generate_during_eval

        if args.padding_free:
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
        self.padding_free = args.padding_free

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomptued_eval_ref_log_probs = False

        # TODO: should add some args regarding to the tricks we apply in computing loss. e.g. whether to clip ref_model logps

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.dataset_num_proc = args.dataset_num_proc

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True


        #############################
        # preprocess data (tokenizing)
        #############################
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        
        if self.ref_model is None:
            if not self.precompute_ref_log_probs:
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))


        self.frozen_model = copy.deepcopy(self.model)
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        self.frozen_model.eval()

    
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DRDPOConfig,
        dataset_name: str,
        ) -> Union[Dataset, IterableDataset]:

        # Build the kwargs for the `map` function
        map_kwargs = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc
        with PartialState().local_main_process_first():

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["prompt", "chosen", "rejected","a1","a2"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset
    
    
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        # TODO: annotate this function
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        a1_input_ids = tokenizer(features["a1"], add_special_tokens=False)["input_ids"]
        a2_input_ids = tokenizer(features["a2"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        a1_input_ids = a1_input_ids + [tokenizer.eos_token_id]
        a2_input_ids = a2_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            a1_input_ids = a1_input_ids[:max_completion_length]
            a2_input_ids = a2_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "a1_input_ids": a1_input_ids,
            "a2_input_ids": a2_input_ids,
        }
    
    @staticmethod
    def process_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        pass

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        pass
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "a1_input_ids",
                "a2_input_ids",
                "ref_a1_logps",
                "ref_a2_logps",
                "a1_preference",
                "rank",
                "image_sizes",
            ]

    
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, 
        batch would be like this:
        prommpt_input_ids: [batch_size, prompt_len]
        prompt_attention_mask: [batch_size, prompt_len]
        compeltion_input_ids: [batch_size, completion_len]
        completion_attention_mask: [batch_size, completion_len]

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        model_kwargs = {}
        # if self.aux_loss_enabled:
        #     model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in batch:
            model_kwargs["pixel_values"] = batch["pixel_values"]
        if "pixel_attention_mask" in batch:
            model_kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"]
        if "image_sizes" in batch:
            model_kwargs["image_sizes"] = batch["image_sizes"]

        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        completion_input_ids = batch["completion_input_ids"]
        completion_attention_mask = batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["a1_logps"] = all_logps[:num_examples]
        output["a2_logps"] = all_logps[num_examples:]


        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the a1 tokens,
            # and the second half to the a2 tokens.
            # To find the start of the a2 tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_a1_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_a2_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()

        else:
            mean_a1_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_a2_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_a1_logits"] = mean_a1_logits
        output["mean_a2_logits"] = mean_a2_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def compute_log_probs(self, batch: dict[str, torch.LongTensor], model):
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compute_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad() and compute_context_manager:
            model_output = self.concatenated_forward(model, batch)
            return model_output["a1_logps"], model_output["a2_logps"]

    
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_model = self.ref_model
        preference_model = self.preference_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        generation_config = self.generation_config
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("=========start training===============")
        start_time = time.time()
        stats_shape =  (args.num_drpo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        

        self.state.global_step = 0
        self.state.episode = 0
        self.max_steps = args.num_total_batches* args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)


        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1* args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["prompt_input_ids"].to(device)
                a1s = data["a1_input_ids"].to(device)
                a2s = data["a2_input_ids"].to(device)
                rank = data["rank"].to(device)

                # Compute the reference log probabilities
                # TODO generate a_star samples

                # TODO init list for stats and metrics

                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params = False
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_responses[:,context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob_star = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    
                    ref_output = forward(ref_model, query_response, processing_class_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7 # TODO: see why this matters
                    ref_logprob = selective_log_softmatx(ref_logits, response) # pad_token_id=0 is required here
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()
                    # FIXME: expand the original batch into number_of_a_star_samples * batch_size batch


            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end  = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, self.pad_token_id):
                    with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        # FIXME: in this seting
                        mb_rank = rank[micro_batch_inds]
                        mb_ref_logprobs = ref_logprobs[micro_batch_inds]
                        mb_pref_a1_a2 = pref_a1_a2[micro_batch_inds]

                        mb_a1 = a1s[micro_batch_inds]
                        mb_query_a1 = query_a1[micro_batch_inds]

                        output_a1, _ = forward(model, mb_query_a1, processing_class.pad_token_id)
                        logits = output_a1.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7 # TODO: need to figure out why
                        model_logprobs = selective_log_softmax(logits, mb_a1)

                        model_logprobs = torch.masked_fill(model_logprobs, padding_mask[micro_batch_inds], 0)

                        importance_sampling_ratio = torch.exp(model_logprobs - mb_ref_logprobs)

                        l1 = - importance_sampling_ratio * (mb_rank - mb_pref_a1_a2)


                        output_a_star, _ = forward(model, mb_query_a_star, processing_class.pad_token_id)
                        logits = output_a_star.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        star_logprobs = selective_log_softmax(logits, mb_a_star)

                        star_logprobs = torch.masked_fill(star_logprobs, padding_mask[micro_batch_inds], 0)# FIXME： padding_mask can be wrong

                        l2 = - torch.sum(star_logprobs * pref_astar_a2, dim = -1) / args.num_star_samples

                        loss = mased_mean(l1 + l2, ~padding_mask[micro_batch_inds])
                        
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        with torch.no_grad():
                            pass # TODO add some stats and logging here

                    gradient_accumulation_idx += 1
                minibatch_idx += 1

                del (
                    output_a1, output_a_star, model_logprobs, star_logprobs, l1, l2, loss,
                    mb_rank, mb_ref_logprobs, mb_pref_a1_a2, mb_a1, mb_query_a1, importance_sampling_ratio,
                    mb_a_star, mb_query_a_star, pref_astar_a2
                )
                torch.cuda.empty_cache()
        
        with torch.no_grad():
            pass # TODO add some metrics

        self.lr_scheduler.step()
        self.control = self.callback_hander.on_step_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        
        # TODO: del metrics
        gc.collect()


    

                        







        

        

                
                


















def get_reward_model(base_causal_model, base_llm_model, value_head_dim: int, add_prompt_head: bool, is_general_preference: bool=False):
    class CustomRewardModel(base_causal_model):

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.is_general_preference = is_general_preference   
            
            self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False) 
            if add_prompt_head:
                self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False)

        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            
            if not self.is_general_preference:
                values = self.value_head(last_hidden_states).squeeze(-1)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
            else:
                values = self.value_head(last_hidden_states)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1, :]
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                    eos_indices = eos_indices.unsqueeze(1)  # Change shape to [batch_size, 1]                  
                    reward_list = []
                    for dim in range(self.value_head.out_features):
                        reward_list.append(values[:,:,dim].gather(dim=1, index=eos_indices))
                    reward = torch.cat(reward_list, dim=1)
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
    return CustomRewardModel




class GPMPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0"), is_general_preference: bool=True, bf16: bool=True, truncation: bool=True, max_length: int=4096, padding: bool=True, tau: float=0.1):
        self.device = device
        self.is_general_preference = is_general_preference

        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.tau = tau
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2" 
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)

        try:
            dir_path = snapshot_download(repo_id=model_name_or_path)
        except Exception as e:
            dir_path = model_name_or_path
        combined_weights = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".safetensors"):
                file_path = os.path.join(dir_path, filename)
                weights = load_file(file_path)
                combined_weights.update(weights)

        if "value_head.weight" in combined_weights:
            self.value_head_dim = combined_weights["value_head.weight"].shape[0]

        self.add_prompt_head = True if "prompt_head.weight" in combined_weights else False

        cls_class = get_reward_model(base_causal_class, base_class, add_prompt_head=self.add_prompt_head, value_head_dim=self.value_head_dim, is_general_preference=is_general_preference)

        # configure model
        self.model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
        )
        
        
        # prepare model
        self.model.to(device)
        self.model.eval()

    def __call__(self, input_id, attention_mask, return_prompt=False):
        # input_texts = [self.tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]
        with torch.no_grad():
            rewards, _ = self.model.custom_forward(input_ids = input_id,attention_mask = attention_mask)

        return rewards   

def generate_high_dim_result(value_head_dim, chosen_reward, rejected_reward):
    R_matrix = torch.zeros((value_head_dim, value_head_dim), device=chosen_reward.device, dtype=chosen_reward.dtype)
    for i in range(0, value_head_dim, 2):
        R_matrix[i, i+1] = -1 
        R_matrix[i+1, i] = 1   
    if chosen_reward.device == rejected_reward.device == R_matrix.device:
        transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
        result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, value_head_dim), rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
        result = result.view(chosen_reward.shape[0])  
    return result

def generate_2_dim_result(chosen_reward, rejected_reward):
    result = chosen_reward[:, 0] * rejected_reward[:, 1] - chosen_reward[:, 1] * rejected_reward[:, 0] 
    return result

def get_a1_preference(a1_input_id, a1_attention_mask, a2_input_id, a2_attention_mask):
    def get_rm(path):
        return GPMPipeline(path)
    rm = get_rm("/home/kyle/Documents/lab/results/gpm_tldr_3e_8dim")
    reward_a1 = rm(a1_input_id, a1_attention_mask)
    reward_a2 = rm(a2_input_id, a2_attention_mask)
    if rm.value_head_dim == 2:
        result_a = generate_2_dim_result(reward_a1, reward_a2)
    else:
        result_a = generate_high_dim_result(rm.value_head_dim, reward_a1, reward_a2)
    p_a = torch.sigmoid(result_a).cpu().detach().numpy().tolist()
    
    return p_a
