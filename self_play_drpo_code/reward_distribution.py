import os
import torch
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,DebertaV2ForSequenceClassification, GPTNeoXForCausalLM
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import DataCollatorWithPadding
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import re
import yaml

LOCAL_TRL_PARENT = "/workspace/Self_play_DRPO"
if LOCAL_TRL_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_TRL_PARENT)

    
# now the import will use your local copy:
from trl import (
    DPOTrainer,
    DPOConfig,
    ModelConfig,
    DRPOTrainer,
    DRPOConfig,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.data_utils import apply_chat_template
data_cache_path = "/workspace/dataset"
model_cache_path = '/workspace/model_cache'
ref_policy_path = 'cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr'
target_policy_path = 'cleanrl/EleutherAI_pythia-1b-deduped__ppo__tldr'
reward_model_path = 'cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr'
ds_path = 'Kyleyee/train_data_tldr'


def load_model(model_path, task = 'generation', model_type = 'decoder', model_cache_path =  '/workspace/model_cache'):

    model_args = ModelConfig(model_path)
    model_torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
    model_kwargs = dict(
    revision = model_args.model_revision,
    torch_dtype = model_torch_dtype, 
    trust_remote_code = model_args.trust_remote_code,
    )

    padding_side = 'left' if model_type == 'decoder' else 'right'
    truncation_side = 'left' if model_type == 'decoder' else 'right'

    if task == 'generation':
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
            cache_dir = model_cache_path,
        )

    elif task == 'reward':
        model_instance = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
            cache_dir = model_cache_path,
        )
    

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side = padding_side, 
        truncation_side = truncation_side,
        use_fast = True,
        trust_remote_code = model_args.trust_remote_code,
        cache_dir = model_cache_path
    )

    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    if getattr(model_instance.config, "pad_token_id", None) is None:
        model_instance.config.pad_token_id = model_tokenizer.pad_token_id

    if model_tokenizer.eos_token is None:
        model_tokenizer.eos_token = model_tokenizer.pad_token  

    if getattr(model_instance.config, "eos_token_id", None) is None:
        model_instance.config.eos_token_id = model_tokenizer.eos_token_id

    return model_instance, model_tokenizer


def generate_response(model_instance, model_tokenizer, prompts, 
                      temperature=0.0, max_new_tokens=256, 
                      n_responses=2, batch_size=8, device='cuda'):
    generation_kwargs = {
        "top_k": 50,
        "top_p": 0.9,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
        "eos_token_id": model_tokenizer.eos_token_id
    }

    if isinstance(prompts, str):
        prompts = [prompts]

    model_instance.to(device)
    all_prompt_responses = []

    # loop over batches
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start:start+batch_size]

        encoded_inputs = model_tokenizer(
            batch_prompts, 
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        prompt_lengths = encoded_inputs.attention_mask.sum(-1).tolist()

        outputs = model_instance.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_responses,
            **generation_kwargs
        )

        generated_tokens = []
        for i in range(len(batch_prompts)):
            prompt_length = prompt_lengths[i]
            for j in range(n_responses):
                idx = i * n_responses + j
                generated_tokens.append(outputs[idx][prompt_length:])

        decoded_responses = model_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        for i, prompt in enumerate(batch_prompts):
            for j in range(n_responses):
                idx = i * n_responses + j
                all_prompt_responses.append([prompt, decoded_responses[idx]])

        del encoded_inputs, outputs, generated_tokens, decoded_responses
        if device == 'cuda':
            torch.cuda.empty_cache()

    return all_prompt_responses



def get_reward_batch(
    model_instance, 
    model_tokenizer, 
    inputs,                    # list[(prompt, response)]
    device='cuda', 
    batch_size=32
):
    model_instance.to(device)
    all_rewards = []

    for start in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[start:start+batch_size]

        # concat prompt+response text
        responses = [prompt + response for (prompt, response) in batch_inputs]

        encoded_inputs = model_tokenizer(
            responses,
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model_instance(**encoded_inputs)

        # Typical reward heads return shape [B] or [B,1] in outputs.logits
        rewards = outputs.logits.squeeze(-1).detach().cpu().tolist()
        if isinstance(rewards, list):
            all_rewards.extend(rewards)
        else:
            all_rewards.append(rewards)

        del encoded_inputs, outputs
        if device == 'cuda':
            torch.cuda.empty_cache()

    return all_rewards





@torch.no_grad()
def get_expected_kl(
    ref_model,
    tokenizer,                 # same tokenizer for both models
    target_model,
    inputs,                    # list[[prompt, response], ...]
    device='cuda',
    batch_size=8,
    max_length=1024,
    response_only=False,        # if True, only score response tokens
):
    ref_model.to(device).eval()
    target_model.to(device).eval()

    seq_diff_all = []

    for start in tqdm(range(0, len(inputs), batch_size)):
        batch_pairs = inputs[start:start+batch_size]
        prompts  = [p for p, r in batch_pairs]
        full_txt = [p + r for p, r in batch_pairs]   # concatenate exactly as specified

        # Encode full sequences (prompt+response)
        enc_full = tokenizer(
            full_txt,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        # Encode prompts alone to get per-sample prompt token counts
        enc_prompt = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        input_ids = enc_full["input_ids"]        # [B, T]
        attn_mask = enc_full["attention_mask"]   # [B, T]
        prompt_len = enc_prompt["attention_mask"].sum(dim=1).to(torch.long)  # [B]

        # Forward both models
        ref_logits    = ref_model(input_ids=input_ids,    attention_mask=attn_mask).logits   # [B, T, V]
        target_logits = target_model(input_ids=input_ids, attention_mask=attn_mask).logits   # [B, T, V]

        # Teacher-forced next-token logprobs
        ref_logp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)      # [B, T-1, V]
        tgt_logp = F.log_softmax(target_logits[:, :-1, :], dim=-1)   # [B, T-1, V]
        labels   = input_ids[:, 1:]                                   # [B, T-1]
        mask     = attn_mask[:, 1:].to(ref_logp.dtype).clone()        # [B, T-1]

        # If we only want the response, drop prompt contributions per sample.
        if response_only:
            # After shifting, tokens with position < (prompt_len - 1) belong to the prompt
            cuts = (prompt_len - 1).clamp_min(0)                      # [B]
            # Zero out up to 'cut' for each row
            # Note: sequences are padded; mask already 0 on pad positions
            for i, cut in enumerate(cuts.tolist()):
                if cut > 0:
                    cut = min(cut, mask.shape[1])  # guard against extreme lengths
                    mask[i, :cut] = 0.0

        # Gather log-prob at the *label* token only (no full-vocab sums)
        ref_tok_lp = ref_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        tgt_tok_lp = tgt_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        tgt_tok_p = tgt_tok_lp.exp()

        diff = (tgt_tok_p * (tgt_tok_lp - ref_tok_lp) * mask).sum(dim = 1)

        seq_diff_all.append(diff.detach().cpu())

        # cleanup
        del enc_full, enc_prompt, ref_logits, target_logits, ref_logp, tgt_logp, labels, mask, ref_tok_lp, tgt_tok_lp
        if device == 'cuda':
            torch.cuda.empty_cache()

    diff_list = torch.cat(seq_diff_all).tolist()

    
    return diff_list



if __name__ == '__main__':
    ref_model_instance, ref_model_tokenizer = load_model(ref_policy_path, task = 'generation')
    target_model_instance, target_model_tokenizer = load_model(target_policy_path, task = 'generation')
    #reward_model_instance, reward_model_tokenizer = load_model(reward_model_path, task ='reward')

    ds = load_dataset(ds_path, cache_dir=data_cache_path, split = 'train')
    prompts = ds['prompt']
    temperature = [0.1, 0.3, 0.5, 0.7, 1.0]
    total = 5000
    step = 10
    prompt = ds['prompt'][0]
    logp_diff_dict = {}
    for t in temperature:
        outputs = []
        for i in tqdm(range(0, total, step)):
            output = generate_response(target_model_instance, target_model_tokenizer, prompt, n_responses=step, temperature = t)
            outputs.extend(output)
        logp_diff = get_expected_kl(
                ref_model_instance, ref_model_tokenizer, target_model_instance, outputs, response_only = True
        )
        logp_diff_dict[f'temperature_{t}'] = logp_diff
    logp_diff_ds = Dataset.from_dict(logp_diff_dict)
    logp_diff_ds.push_to_hub('dpo_reward_dist_pi_theta')
    