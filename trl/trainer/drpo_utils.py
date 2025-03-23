from typing import Optional, List, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, pipeline
import torch.nn.functional as F
from transformers import AutoTokenizer     
import os
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from datasets import load_dataset
import re
from dataclasses import dataclass, field
from typing import Optional
from datasets import  DatasetDict
from datasets import load_dataset
from huggingface_hub import ModelCard


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

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
        
        def create_skew_symmetric_block_matrix(self, dim, device, dtype, prompt_hidden_states):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states. Only the relevant block diagonal parts are generated.
            
            Args:
            - dim: Dimension of the square matrix (must be even).
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim].
            
            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries.
            """
            if hasattr(self, 'prompt_head'):
                batch_size = prompt_hidden_states.shape[0]
                
                # Ensure that dim is even, as we're creating blocks of size 2x2
                assert dim % 2 == 0, "dim must be even for skew-symmetric block generation"

                # Pass through the linear layer to get the block diagonal entries (half of the matrix's off-diagonal blocks)
                block_values = self.prompt_head(prompt_hidden_states).view(batch_size, dim // 2)
                block_values = torch.softmax(block_values, dim=-1)
                
                # Create a batch of zero matrices [batch_size, dim, dim]
                batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
                
                # Fill only the block diagonal entries with the learned values
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                    batch_R_matrices[:, i + 1, i] = block_values[:, i // 2]  # Skew-symmetric condition
            else:
                raise AttributeError("prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization.")
                
            return batch_R_matrices
         
    return CustomRewardModel

class GPMPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_general_preference: bool=True, bf16: bool=True, truncation: bool=True, max_length: int=4096, padding: bool=True, tau: float=0.1):
        self.device = device
        self.is_general_preference = is_general_preference

        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.tau = tau
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager" 
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)

        try:
            dir_path = snapshot_download(repo_id=model_name_or_path)
            # print(dir_path)
        except Exception as e:
            dir_path = model_name_or_path
        combined_weights = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".safetensors"):
                file_path = os.path.join(dir_path, filename)
                weights = load_file(file_path)
                combined_weights.update(weights)

        # print(combined_weights.keys())

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
            device_map="auto",
            attn_implementation="eager"
        )
        
        # configure tokenizer
        self.tokenizer = get_tokenizer(model_name_or_path, self.model, "left", use_fast=True)
        self.tokenizer.truncation_side = "right"
        
        # prepare model
        self.model.to(device)
        self.model.eval()

    def __call__(self, samples: List[str], return_prompt=False):
        input_texts = samples

        print("++++++++++++++++++++++\n length of inputs of preference model")
        print(len(input_texts))
        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        print("end of tokenizer")
        

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        

        inputs["input_ids"][:, -1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][:, -1] = 1

        with torch.no_grad():
            rewards, _ = self.model.custom_forward(**inputs, return_output=return_prompt)

        return rewards
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self




def get_preference_score(preference_model, a_1_iuput, a_2_input, is_bt_model:bool = True):
    # print(a_1_iuput)
    # preference_model = GPMPipeline("Kyleyee/gpm_tldr_3e")
    a_1_reward = preference_model(a_1_iuput)
    a_2_reward = preference_model(a_2_input)
    if is_bt_model:
        result = a_1_reward - a_2_reward
    else:
        if preference_model.value_head_dim == 2:
            result = a_1_reward[:, 0] * a_2_reward[:, 1] - a_1_reward[:, 1] * a_2_reward[:, 0]
        else:
            R_matrix = torch.zeros((preference_model.value_head_dim, preference_model.value_head_dim), device=a_1_reward.device, dtype=a_1_reward.dtype)
            for i in range(0, preference_model.value_head_dim, 2):
                R_matrix[i, i+1] = -1 
                R_matrix[i+1, i] = 1   
            if a_1_reward.device == a_2_reward.device == R_matrix.device:
                transformed_a_1 = torch.matmul(a_1_reward, R_matrix.T)
                result = torch.bmm(transformed_a_1.view(a_1_reward.shape[0], 1, preference_model.value_head_dim), a_2_reward.view(a_2_reward.shape[0], preference_model.value_head_dim, 1))
                result = result.view(a_1_reward.shape[0])  
    p = F.sigmoid(result)
    return p


class BTPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), truncation: bool=True, padding: bool=True):
        self.pipeline = pipeline("sentiment-analysis", model=model_name_or_path, device=device)
        self.truncation=truncation
        self.padding=padding

    def __call__(self, input_text: List[str]):
        batch_size = len(input_text)
        sentiment_results = self.pipeline(input_text, batch_size=batch_size, truncation=self.truncation, padding=self.padding)
        return torch.tensor([res["score"] if res["label"] == "POSITIVE" else 1 - res["score"] for res in sentiment_results])
    
    def to(self, device):
        self.device = device
        self.pipeline.model.to(device)
        return self 