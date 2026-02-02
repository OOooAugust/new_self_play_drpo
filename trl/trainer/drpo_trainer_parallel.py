from typing import Any, Union, Optional, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import (
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import seed_worker
from .drpo_config import DRPOConfig
from ..models.utils import unwrap_model_for_generation
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from .utils import (
    pad,
    truncate_right,
    selective_log_softmax,
    prepare_deepspeed
)
from .drpo_utils import get_preference_score
from dataclasses import dataclass
from accelerate import PartialState


@dataclass
class DataCollatorDRPO(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = 'pt'

    def torch_call(self, examples: list[Union[list[int], dict[str,Any], Any]])-> dict[str, Any]:
        # print(examples)
        prompt_ids = [torch.tensor(example["prompt_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(id) for id in prompt_ids]
        a1_ids = [torch.tensor(example["a1_ids"]) for example in examples]
        a2_ids = [torch.tensor(example["a2_ids"]) for example in examples]
        ref_ids = [torch.tensor(example["ref_ids"]) for example in examples]
        a1_attention_mask = [torch.ones_like(id) for id in a1_ids]
        a2_attention_mask = [torch.ones_like(id) for id in a2_ids]
        ref_attention_mask = [torch.ones_like(id) for id in ref_ids]

        output = {}
        output["prompt_ids"] = pad(prompt_ids, padding_value = self.pad_token_id, padding_side = "left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value = 0, padding_side = "left")
        output["a1_ids"] = pad(a1_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a1_attention_mask"] = pad(a1_attention_mask, padding_value = 0, padding_side = "right")
        output["a2_ids"] = pad(a2_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a2_attention_mask"] = pad(a2_attention_mask, padding_value = 0, padding_side = "right")
        output["ref_ids"] = pad(ref_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["ref_attention_mask"] = pad(ref_attention_mask, padding_value = 0, padding_side = "right")

        if "preference_score" in examples[0]:
            output["preference_score"] = torch.tensor([example["preference_score"] for example in examples])

        output["rank"] = torch.tensor([example["rank"] for example in examples])
        return output


def is_conversational(example: dict[str, Any]) -> bool:
    for key in ("prompt", "a1", "a2"):
        value = example.get(key)
        if not isinstance(value, list) or not value:
            continue
        first_msg = value[0]
        if isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg:
            return True
    return False



def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, str]:
    
    if not all(key in example for key in ["prompt", "a1", "a2"]):
        raise KeyError(f"Example must contain 'prompt', 'a1', and 'a2' keys. Got: {list(example.keys())}")


    last_role = example["prompt"][-1]["role"]
    if last_role == "user":
        add_generation_prompt = True
        continue_final_message = False
    elif last_role == "assistant":
        add_generation_prompt = False
        continue_final_message = True
    else:
        raise ValueError(f"Invalid role in the last message: {last_role}")

    prompt = tokenizer.apply_chat_template(
        example["prompt"],
        continue_final_message=continue_final_message,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


    def extract_completion(response_key: str) -> str:
        full_text = tokenizer.apply_chat_template(
            example["prompt"] + example[response_key], tokenize=False
        )
        if not full_text.startswith(prompt):
            raise ValueError(
                f"Chat template for prompt + {response_key} does not start with prompt. "
                "This may indicate an unsupported chat template."
            )
        return full_text[len(prompt):]

    return {
        "prompt": prompt,
        "a1": extract_completion("a1"),
        "a2": extract_completion("a2"),
    }


def maybe_apply_chat_template(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, str]:
    if is_conversational(example):
        return apply_chat_template(example, tokenizer)
    return example


class DRPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Union[PreTrainedModel, nn.Module],
        preference_model: Union[PreTrainedModel, nn.Module],
        args: DRPOConfig,
        dpo_as_reward: False, 
        dpo_model: Union[PreTrainedModel, nn.Module, None] = None,
        train_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollatorDRPO] = None, 
    ) -> None:
        
        if ref_model is model:
            raise ValueError('reference model and target model cannot be the same')
        self.ref_model = ref_model
        self.ref_model.eval()

        if preference_model is None:
            raise ValueError('preference model cannot be None')
        self.preference_model = preference_model
        self.preference_model.eval()  #double check, do I need to train preference model in the training step?

        if dpo_as_reward and dpo_model is None: 
            raise ValueError('dpo model cannot be None if use dpo as reward')
        if dpo_as_reward: 
            self.dpo_model = dpo_model
            self.dpo_model.eval()

        if data_collator is None:
            if processing_class is None:
                raise ValueError('processing class should be provided if no data collator')
            data_collator = DataCollatorDRPO(pad_token_id = processing_class.pad_token_id)
        

        self.processing_class = processing_class

        train_dataset = self._prepare_dataset(train_dataset, processing_class, args)
        self.generation_config = GenerationConfig(
            max_new_tokens = args.max_new_tokens, 
            temperature = args.generate_temperature, 
            top_k = 50 if args.generate_temperature > 0 else None, 
            top_p = 0.1 if args.generate_temperature > 0 else None, 
            do_sample = True if args.generate_temperature > 0 else False,
            use_cache=False if args.gradient_checkpointing else True
        )
        

        super().__init__(
            model = model, 
            args = args,
            data_collator = data_collator, 
            train_dataset = train_dataset,
            processing_class = processing_class
        )

        if self.args.learn_mu_parameters:
            model.mu_head = nn.ModuleDict()
            if 'mu_ref' not in model.mu_head:
                model.mu_head['mu_ref'] = self._build_mu_value_network(3, self.args.mu_head_hidden_size)
            if 'mu_theta' not in model.mu_head:
                model.mu_head['mu_theta'] = self._build_mu_value_network(3, self.args.mu_head_hidden_size)

        if self.is_deepspeed_enabled:
            if self.preference_model is not None:
                self.preference_model = prepare_deepspeed(
                    self.preference_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.ref_model is not None:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.dpo_model is not None:
                self.dpo_model = prepare_deepspeed(
                    self.dpo_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.preference_model is not None:
                self.preference_model = self.preference_model.to(self.accelerator.device)
            if self.dpo_model is not None:
                self.dpo_model = self.dpo_model.to(self.accelerator.device)

        
        self.stats = {}
        self.stats['beta'] = []
        self.stats['objective/kl'] = []
        if self.args.loss_type == 'IS':
            self.stats["objective/IS_loss"] = []
            self.stats["logps/a1"] = []
            self.stats["logps/a1_ref"] = []
            self.stats["ps/a1"] = []
            self.stats["logps/a*"] = []
            self.stats["logps/a*_ref"] = []
            self.stats["ps/a*"] = []
            self.stats["is_ratio"] = []
            self.stats["clipped_ratio"] = []

            
    @staticmethod
    def tokenize_row(example: dict[str, Any],
                     processing_class: PreTrainedTokenizerBase,
                     max_prompt_length: Union[int, None] = None, 
                     max_completion_length: Union[int, None] = None,
                     add_special_tokens_for_prompt: bool = True,
                     eos_after_completion: bool = True) -> dict[str, Any]:
        
        prompt_ids = processing_class(example['prompt'], add_special_tokens = False)['input_ids']
        a1_ids = processing_class(example['a1'], add_special_tokens = False)['input_ids']
        a2_ids = processing_class(example['a2'], add_special_tokens = False)['input_ids']
        ref_ids = processing_class(example['ref'], add_special_tokens = False)['input_ids']


        if add_special_tokens_for_prompt:
            if processing_class.bos_token_id is not None:
                prompt_ids = [processing_class.bos_token_id] + prompt_ids
            if processing_class.eos_token_id is not None:
                prompt_ids = prompt_ids + [processing_class.eos_token_id]

        if eos_after_completion:
            a1_ids = a1_ids + [processing_class.eos_token_id]
            a2_ids = a2_ids + [processing_class.eos_token_id]
            ref_ids = ref_ids + [processing_class.eos_token_id]
        
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[-max_prompt_length:]
        if max_completion_length is not None:
            a1_ids = a1_ids[:max_completion_length]
            a2_ids = a2_ids[:max_completion_length]
            ref_ids = ref_ids[:max_completion_length]
        
        return {
            'prompt_ids':prompt_ids, 
            'a1_ids':a1_ids, 
            'a2_ids':a2_ids,
            'ref_ids':ref_ids
        }
    
    def _set_signature_columns_if_needed(self):

        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids",
                "a1_ids",
                "a2_ids",
                "prompt_attention_mask",
                "a1_attention_mask",
                "a2_attention_mask",
                "rank",
                'ref_ids',
                'ref_attention_mask'
            ]

    
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
    






    def _prepare_dataset(self,
                         dataset: Union[Dataset, IterableDataset],
                         processing_class: PreTrainedTokenizerBase,
                         args: DRPOConfig) -> Union[Dataset, IterableDataset]:
        
        map_kwargs = {'writer_batch_size': 10}
        if isinstance(dataset, Dataset):
            map_kwargs['num_proc'] = args.dataset_num_proc

        with PartialState().local_main_process_first():
            # Step 1: Apply chat template if data is in conversational format
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={'tokenizer': processing_class},
                **map_kwargs
            )

            # Step 2: Tokenize the dataset
            dataset = dataset.map(
                self.tokenize_row, 
                remove_columns=['prompt', 'a1', 'a2', 'ref'],
                fn_kwargs={
                    'processing_class': processing_class,
                    'max_prompt_length': args.max_prompt_length,
                    'max_completion_length': args.max_completion_length,
                    'add_special_tokens_for_prompt': False, 
                    'eos_after_completion': args.eos_after_completion
                },
                **map_kwargs
            )
        return dataset 
    

    def _generate(
        self, 
        model,
        prompt_ids: torch.tensor,
        prompt_attention_mask: torch.tensor, 
        num_astar: int = 1
    ):

        prompt_ids = prompt_ids.repeat(num_astar, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_astar, 1)


        with unwrap_model_for_generation(model, self.accelerator, gather_deepspeed3_params=False) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids = prompt_ids, 
                attention_mask = prompt_attention_mask, 
                generation_config = self.generation_config,
                pad_token_id = self.processing_class.pad_token_id,
                eos_token_id = self.processing_class.eos_token_id 
            )
        
        completion_ids = output[:, prompt_ids.shape[1]:]
        completion_ids, completion_attention_mask = truncate_right(completion_ids, self.processing_class.eos_token_id , self.processing_class.pad_token_id)
        
        return prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask

    def _forward(
        self,
        model, 
        prompt_ids, 
        prompt_attention_mask,
        completion_ids, 
        completion_attention_mask
    ):
        num_tokens_to_trauncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.args.max_length, 0)
        prompt_ids = prompt_ids[:, num_tokens_to_trauncate:]
        prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_trauncate:]
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        logits = output.logits[:, max(0, prompt_ids.size(1) - 1) : -1]

        if self.args.forward_temperature > 0:
            logits /= self.args.forward_temperature + 1e-7

        if completion_ids.size(1) > logits.size(1):
            completion_ids = completion_ids[:, :logits.size(1)]
        
        logps = selective_log_softmax(logits, completion_ids)
        del (output, logits)
        return logps 
    

    def _calculate_kl_divergence(
        self,
        target_model, 
        ref_model, 
        prompt_ids, 
        prompt_attention_mask, 
        completion_ids, 
        completion_attention_mask
    ):
        
        #full distribution KL divergence 
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.args.max_length, 0)
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_truncate:]
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        logits_target = target_model(prompt_completion_ids, attention_mask=prompt_completion_mask).logits
        with torch.no_grad():
            logits_ref = ref_model(prompt_completion_ids, attention_mask=prompt_completion_mask).logits
        prompt_length = prompt_ids.size(1)
        logits_completion_target = logits_target[:, max(0, prompt_length - 1) : -1, :]
        logits_completion_ref = logits_ref[:, max(0, prompt_length - 1) : -1, :]
        
        logp_completion_target = F.log_softmax(logits_completion_target, dim=-1)
        logp_completion_ref = F.log_softmax(logits_completion_ref, dim=-1)

        kl_per_token = F.kl_div(
            input=logp_completion_ref,         
            target=logp_completion_target,        
            log_target=True,
            reduction="none",
        ).sum(-1)

        L_logit = logits_completion_target.size(1)                 
        drop = 0 if prompt_length > 0 else 1                      
        mask = completion_attention_mask[:, drop:drop + L_logit]  
        mask = mask.to(logits_completion_target.dtype)  

        kl = (kl_per_token * mask).sum(-1) 

        del (logits_target, logits_ref, logits_completion_target, logits_completion_ref, logp_completion_target, logp_completion_ref, mask)


        return kl
    

    def _build_mu_value_network(self, input_dim: int, hidden_size: int) -> nn.Module:
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

    def training_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        
        model.train()
        
        prompt_ids = inputs['prompt_ids']
        batch_size = prompt_ids.size(0)
        prompt_attention_mask = inputs['prompt_attention_mask']
        a1_ids = inputs['a1_ids']
        a1_attention_mask = inputs['a1_attention_mask']
        a2_ids = inputs['a2_ids']
        a2_attention_mask = inputs['a2_attention_mask']
        ref_ids = inputs['ref_ids']
        ref_attention_mask = inputs['ref_attention_mask']
        rank = inputs['rank'].float()
        

        per_token_logps = self._forward(model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask)
        prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask = self._generate(model, prompt_ids, prompt_attention_mask, self.args.num_astar)
        per_token_logps_star = self._forward(model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask)

        with torch.no_grad():
            per_token_ref_logps = self._forward(self.ref_model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask)
            per_token_ref_logps_star = self._forward(self.ref_model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask)
        
        
        prompt_astar_ids = torch.cat((prompt_ids_repeated, astar_ids), dim=1)
        prompt_a1_ids = torch.cat((prompt_ids, a1_ids), dim=1)
        prompt_a2_ids = torch.cat((prompt_ids, a2_ids), dim=1)

        prompt_astar = self.processing_class.batch_decode(prompt_astar_ids, skip_special_tokens=True)
        #prompt_a1 = self.processing_class.batch_decode(prompt_a1_ids, skip_special_tokens=True)
        prompt_a2 = self.processing_class.batch_decode(prompt_a2_ids, skip_special_tokens=True)

        prompt_a2_repeated = prompt_a2 * self.args.num_astar
        
        with torch.inference_mode():

            preference_score_star= get_preference_score(
                self.preference_model, 
                prompt_astar, 
                prompt_a2_repeated,
                is_bt_model = self.args.is_bt_model,
                noisy = 0.2,
                kwargs=self.args.preference_model_kwargs or {}
            )
            """
            preference_score = get_preference_score(
                self.preference_model, 
                prompt_a1, 
                prompt_a2,
                is_bt_model = self.args.is_bt_model,
                noisy = 0.2,
                kwargs = self.args.preference_model_kwargs or {}
            )
            """

        logps_star = (per_token_logps_star * astar_attention_mask).sum(-1)

        

        kl_div = ((torch.exp(per_token_ref_logps_star - per_token_logps_star) - (per_token_ref_logps_star - per_token_logps_star) - 1)*astar_attention_mask).sum(-1)
        mean_kl = kl_div.mean()
        logps = (per_token_logps * a1_attention_mask).sum(1)
        ref_logps = (per_token_ref_logps * a1_attention_mask).sum(1)
        loss = 0 

        if self.args.ratio_processing == 'clip':
            ratio = torch.exp((logps - ref_logps))
        elif self.args.ratio_processing == 'KL_divergence':

            with torch.no_grad():

                per_token_dpo_logps_star = self._forward(
                    self.dpo_model, 
                    prompt_ids_repeated, 
                    prompt_attention_mask_repeated, 
                    astar_ids, 
                    astar_attention_mask
                )

                dpo_logps_star = (per_token_dpo_logps_star * astar_attention_mask).sum(-1)
                ref_logps_star = (per_token_ref_logps_star * astar_attention_mask).sum(-1)
                reward = self.args.dpo_beta * (dpo_logps_star.detach() - ref_logps_star.detach())


            if self.args.learn_mu_parameters:
                mu_total_loss = None
                mu_ref_loss = None 
                mu_theta_loss = None
                mu_ref = None
                mu_theta = None
            else:
                with torch.no_grad():
                    kl_ref_theta_star = self._calculate_kl_divergence(
                        self.ref_model, self.dpo_model,
                        prompt_ids, prompt_attention_mask,
                        ref_ids, ref_attention_mask)

                    kl_theta_ref = self._calculate_kl_divergence(
                        model, self.ref_model,
                        prompt_ids_repeated, prompt_attention_mask_repeated,
                        astar_ids, astar_attention_mask)

                    kl_theta_theta_star = self._calculate_kl_divergence(
                        model, self.dpo_model,
                        prompt_ids_repeated, prompt_attention_mask_repeated,
                        astar_ids, astar_attention_mask)

                    mu_ref = -self.args.dpo_beta * kl_ref_theta_star
                    mu_theta = self.args.dpo_beta * (kl_theta_ref - kl_theta_theta_star)

                if self.args.ratio_distribution == 'normal_distribution':
                    ratio = torch.exp((-1/(2*self.args.scale*self.args.scale))*((reward - mu_ref)**2 - (reward - mu_theta)**2))
                elif self.args.ratio_distribution == 'laplace_distribution':
                    ratio = torch.exp((-1/(2*self.args.scale))*(torch.abs(reward - mu_theta) - torch.abs(reward - mu_ref))) 
                else:
                    raise ValueError('ratio distribution not defined')
                
                del ref_ids, ref_attention_mask, per_token_dpo_logps_star, dpo_logps_star, ref_logps_star
                

        clipped_ratio = torch.clamp(ratio, min = 1. / self.args.clipbound, max = self.args.clipbound)

        if self.args.loss_type == 'IS':
            losses1 = -clipped_ratio.detach() * rank.detach() * logps
        
        loss += losses1.mean() + self.args.beta * mean_kl
        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        #logging
        self.stats['beta'].append(self.args.beta)
        self.stats['objective/kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        if self.args.loss_type == 'IS':
            self.stats['objective/IS_loss'].append(self.accelerator.gather_for_metrics(losses1).mean().item())
            self.stats['logps/a*'].append(self.accelerator.gather_for_metrics(logps_star).mean().item())
            self.stats['logps/a*_ref'].append(self.accelerator.gather_for_metrics(per_token_ref_logps_star*astar_attention_mask).sum(-1).mean().item())
            self.stats['ps/a*'].append(self.accelerator.gather_for_metrics(preference_score_star).mean().item()) 
        self.stats["logps/a1"].append(self.accelerator.gather_for_metrics(logps).mean().item())
        self.stats['logps/a1_ref'].append(self.accelerator.gather_for_metrics(ref_logps).mean().item())
        #self.stats['ps/a1'].append(self.accelerator.gather_for_metrics(preference_score).mean().item()) 
        self.stats['is_ratio'].append(self.accelerator.gather_for_metrics(ratio.mean()).mean().item())

        return loss.detach() 









        
    









            
        

        



        



        





