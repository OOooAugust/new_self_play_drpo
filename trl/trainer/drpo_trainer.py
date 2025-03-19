import os
import textwrap
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union, Optional, Callable, List, Tuple
from functools import wraps 

import transformers
import torch.utils.data



from torch.utils.data import DataLoader, IterableDataset

from transformers import (
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)

from transformers.trainer_utils import seed_worker
from transformers.data.data_collator import DataCollatorMixin

from datasets import Dataset


from ..data_utils import maybe_apply_chat_template
from .utils import unwrap_model_for_generation, pad, truncate_right

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .online_dpo_config import DRPOConfig
from .utils import (
    SIMPLE_CHAT_TEMPLATE,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    prepare_deepspeed,
    truncate_right,
)
from .drpo_utils import get_preference_score # TODO: write get_preference_score function


# if is_peft_available():
#     from peft import PeftModel, get_peft_model

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

class DataCollatorDRPO(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = 'pt'

    def torch_call(self, examples: list[Union[list[int], dict[str,Any], Any]])-> dict[str, Any]:

        prompt_ids = [torch.tensor(example["prompt_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(id) for id in prompt_ids]
        a1_ids = [torch.tensor(example["a1_ids"]) for example in examples]
        a2_ids = [torch.tensor(example["a2_ids"]) for example in examples]
        a1_attention_mask = [torch.ones_like(id) for id in a1_ids]
        a2_attention_mask = [torch.ones_like(id) for id in a2_ids]

        output = {}
        output["prompt_ids"] = pad(prompt_ids, padding_value = self.pad_token_id, padding_side = "left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value = 0, padding_side = "left")
        output["a1_ids"] = pad(a1_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a1_attention_mask"] = pad(a1_attention_mask, padding_value = 0, padding_side = "right")
        output["a2_ids"] = pad(a2_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a2_attention_mask"] = pad(a2_attention_mask, padding_value = 0, padding_side = "right")

        if "preference_score" in examples[0]:
            output["preference_score"] = torch.tensor([example["preference_score"] for example in examples])

        output["rank"] = torch.tensor([example["rank"] for example in examples])

        return output
            


class DRPOTrainer(Trainer):
    
    """
    Initialize the DRPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`): 
            The model to be trained, preferably an `AutoModelForCausalLM`.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module`): 
            The reference model to be used for the KL divergence term.
        preference_model (`transformers.PreTrainedModel` or `torch.nn.Module`): 
            The preference model to be used for the preference score term.
        args (`DRPOConfig`): 
            The training arguments.
        data_collator (`DataCollator`): 
            The data collator to be used for training. defaults to `DataCollatorDRPO`.
        train_dataset (`datasets.Dataset`)
        eval_dataset (`datasets.Dataset`)
        processing_class (`ProcessingClass`): 
            The processing class to be used for tokenization.
            If provided, will be used to automatically process the inputs for the model, 
            and it will be saved along the model to make it easier to rerun an interrupted training or reuse the fine-tuned model.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction`
            and return a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
    """

    _tag_names = ["trl", "drpo"]

    def __init__(
            self,
            model: PreTrainedModel,
            ref_model: Union[PreTrainedModel, nn.Module],
            preference_model: Union[PreTrainedModel, nn.Module],
            args: DRPOConfig,
            data_collator: Optional[DataCollatorDRPO] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = None
        ) -> None:
        
        if ref_model is model:
            raise ValueError("The reference model cannot be the same as the model.")         
        self.ref_model = ref_model
        self.ref_model.eval()

        if preference_model is None:
            raise ValueError("The preference model cannot be None.")
        self.preference_model = preference_model

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        if data_collator is None:
            data_collator = DataCollatorDRPO(pad_token_id=processing_class.pad_token_id)

        self.max_length = args.max_length

        self.stats = {
            "logps/a1": [],
            "logps/a*": [],
            "ps/a1": [],
            "ps/a*": [],
            "beta": [],
            "objective/kl": [],
            "objective/loss1": [],
            "objective/loss2": [],
            "objective/loss": []
        }

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta
        self.ref_model = self.ref_model.to(self.accelerator.device)
        self.preference_model = self.preference_model.to(self.accelerator.device)

    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, 
                     tokenizer: PreTrainedTokenizerBase, 
                     max_prompt_length: Union[int, None] = None, 
                     max_completion_length: Union[int, None] = None, 
                     add_special_token: bool = True) -> dict[str, Any]:
        """Tokenize a row of data."""
        # FIXME: the logic of whether to add special tokens is not clear
        # FIXME: whether to add attention mask is not clear
        
        prompt_ids = tokenizer(feature["prompt"], add_special_tokens=False)["input_ids"]
        a1_ids = tokenizer(feature["a1"], add_special_tokens=False)["input_ids"]
        a2_ids = tokenizer(feature["a2"], add_special_tokens=False)["input_ids"]

        # add speical tokens
        if add_special_token:
            if tokenizer.bos_token_id is not None:
                prompt_ids = [tokenizer.bos_token_id] + prompt_ids
            if tokenizer.eos_token_id is not None:
                prompt_ids = prompt_ids + [tokenizer.eos_token_id]
        
        # 2 completions must add eos token to avoid non-stopping generation
        a1_ids = a1_ids + [tokenizer.eos_token_id]
        a2_ids = a2_ids + [tokenizer.eos_token_id]

        # truncation
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[-max_prompt_length:] #right truncation
        if max_completion_length is not None:
            a1_ids = a1_ids[:max_completion_length]
            a2_ids = a2_ids[:max_completion_length] # left truncation

        return {
            "prompt_ids": prompt_ids,
            "a1_ids": a1_ids,
            "a2_ids": a2_ids
        }
    
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    

    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    
    def _generate(self, model, prompt_ids: torch.tensor, prompt_attention_mask: torch.tensor, num_astar:int = 1):
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # inputs = [{"prompt": prompt} for prompt in prompts]
        # inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        # inputs = [self.tokenize_row(x, self.processing_class) for x in inputs]
        # inputs = self.data_collator(inputs)
        
        # inputs = self._prepare_inputs(inputs)
        prompt_ids = prompt_ids.repeat(num_astar, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_astar, 1)
        with unwrap_model_for_generation(model, self.accelerator, gather_deepspeed3_params=False) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                generation_config = self.generation_config,
            )

        completion_ids = output[:, prompt_ids.shape[1]:]
        completion_ids, completion_attention_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

        return prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask
    
    def _forward(self, model, prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, prompt_ids.size(1) - 1 : -1]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs
    
    def training_step(self, model:nn.Modules, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int]=None) -> torch.Tensor:
        model.train()
        print(inputs)
        batch_size = inputs["prompt_ids"].size(0)

        prompt_ids = inputs["prompt_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        a1_ids = inputs["a1_ids"]
        a1_attention_mask = inputs["a1_attention_mask"]
        a2_ids = inputs["a2_ids"]
        a2_attention_mask = inputs["a2_attention_mask"]
        rank = inputs["rank"]

        # log pi(y|x) shape(batch_size, 1)
        logprobs = self._forward(model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask)

        # sample y* for `num_astar` times
        prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask = self._generate(model, prompt_ids, prompt_attention_mask, self.args.num_astar)
        contain_eos_token = torch.any(astar_ids == self.processing_class.eos_token_id, dim=-1)

        # log pi(y*|x) shape(num_astar*batch_size, 1)
        logprobs_star = self._forward(model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask)
        
        with torch.no_grad():
            if self.ref_model is not None:
                # log pi_ref(y|x)
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask)
                ref_logprobs_star = self._forward(self.ref_model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask)
            else:
                raise NotImplementedError("Peft is not implemented yet and ref model should be specified.")

        device = logprobs.device

        # Compute preference score g(y*, y', x) and g(y, y', x)
        with torch.inference_mode():
            context_length = prompt_ids.size(1)

            prompt_astar_ids = torch.cat((prompt_ids_repeated, astar_ids), dim=1)
            prompt_a2_ids = torch.cat((prompt_ids, a2_ids), dim=1)

            prompt_astar = self.processing_class.batch_decode(prompt_astar_ids, skip_special_tokens=True)
            prompt_a2 = self.processing_class.batch_decode(prompt_a2_ids, skip_special_tokens=True)
            prompt_a2_repeated = prompt_a2 * self.args.num_astar
            assert(len(prompt_astar) == len(prompt_a2_repeated))
            
            # g(y*, y', x)
            preference_score_star= get_preference_score(
                self.preference_model, 
                prompt_astar, 
                prompt_a2_repeated
            )
            
            if self.args.missing_eos_penalty is not None:
                preference_score_star[~contain_eos_token] -= self.args.missing_eos_penalty

            
            if not self.precompute_preference_score:
                # g(y, y', x)            
                prompt_a1_ids = torch.cat((prompt_ids, a1_ids), dim=1)
                prompt_a2_ids = torch.cat((prompt_ids, a2_ids), dim=1)

                prompt_a1 = self.processing_class.batch_decode(prompt_a1_ids, skip_special_tokens=True)
                assert(len(prompt_a1) == len(prompt_a2))

                # TODO: write get_preference_score function
                preference_score = get_preference_score(
                    self.preference_model, 
                    prompt_a1, 
                    prompt_a2
                )
            else:
                preference_score = inputs["preference_score"]

        # Compute the loss part one
        logprobs_sum = (logprobs * a1_attention_mask).sum(1)
        ref_logprobs_sum = (ref_logprobs * a1_attention_mask).sum(1)
        losses1 = - torch.exp(logprobs_sum - ref_logprobs_sum)*(rank - preference_score)

        # Compute the loss part two
        assert logprobs_star.size(0) == batch_size * self.args.num_astar
        logprobs_star = logprobs_star.view(self.args.num_astar, batch_size, -1)
        astar_attention_mask = astar_attention_mask.view(self.args.num_astar, batch_size, -1)
        # take mean over num_astar
        logprobs_star_sum = (logprobs_star * astar_attention_mask).sum(-1).mean(0)

        preference_score_star = preference_score_star.view(self.args.num_astar, -1).mean(0)
        losses2 = - logprobs_star_sum * preference_score_star

        # Compute the penalty term of kl divergence
        # kl_onpolicy_part = ((logprobs_star - ref_logprobs_star)*astar_attention_mask).sum(-1) 
        kl_offline_part = ((logprobs - ref_logprobs)*a1_attention_mask).sum(-1)
        # mean_kl = torch.cat((kl_onpolicy_part, kl_offline_part), dim=0).mean()
        mean_kl = kl_offline_part.mean()

        # Compute the loss
        loss = (losses1 + losses2).mean() - self.beta * mean_kl

        # log everything
        self.stats["logps/a1"].append(self.accelerator.gather_for_metrics(logprobs_sum).mean().item())
        self.stats['logps/a*'].append(self.accelerator.gather_for_metrics(logprobs_star_sum).mean().item())
        self.stats['ps/a1'].append(self.accelerator.gather_for_metrics(preference_score).mean().item()) # preference score
        self.stats['ps/a*'].append(self.accelerator.gather_for_metrics(preference_score_star).mean().item()) # preference score
        self.stats['beta'].append(self.beta)
        self.stats['objective/kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self.stats['objective/loss1'].append(self.accelerator.gather_for_metrics(losses1.mean()).mean().item())
        self.stats['objective/loss2'].append(self.accelerator.gather_for_metrics(losses2.mean()).mean().item())
        self.stats['objective/loss'].append(self.accelerator.gather_for_metrics(loss).mean().item())

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""C Shi, K Ye, J Zhu, H Zhou, E Xu""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Online DPO",
            trainer_citation=citation,
            paper_id="not available",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))



        






            

        
            
        
            
            







        




    
        

            