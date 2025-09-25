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
import numpy as np
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
from trl.trainer.utils import pad, truncate_right, selective_log_softmax
from trl.data_utils import apply_chat_template

data_cache_path = "/workspace/dataset"
model_cache_path = '/workspace/model_cache'
ref_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 
#target_policy_path = "Qwen/Qwen2.5-1.5B-Instruct" 
target_policy_400_path = 'august66/hh_qwen1.5_drpo_400'
target_policy_800_path = 'august66/hh_qwen1.5_drpo_800'
dpo_policy_path = 'august66/hh_qwen_1.5b_dpo_model_2'
#reward_model_path = 'cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr'
ds_path = 'august66/drpo_hh_qwen2.5_1.5b'


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


def generate_responses(
    model_instance,
    model_tokenizer,
    prompts,                      # single chat or list of chats (each: list[{role,content}])
    *,
    temperature=1.0,
    max_new_tokens=256,
    n_responses_total=1000,       # total per prompt
    responses_per_call=16,        # per subcall per prompt
    batch_size=4,                 # prompts per batch
    device='cuda',
    max_length=1024,
):
    # tokenizer prefs
    model_tokenizer.padding_side = "left"
    if hasattr(model_tokenizer, "truncation_side"):
        model_tokenizer.truncation_side = "left"
    if model_tokenizer.pad_token_id is None and model_tokenizer.eos_token_id is not None:
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

    gen_kwargs = {
        "do_sample": bool(temperature > 0),
        "temperature": float(temperature),
        "top_k": 50,
        "top_p": 0.9,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": model_tokenizer.eos_token_id,
        "pad_token_id": model_tokenizer.pad_token_id,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    # normalize prompts input
    if isinstance(prompts, str):
        prompts = [prompts]
    elif (isinstance(prompts, list) and prompts
          and isinstance(prompts[0], dict) and "role" in prompts[0]):
        prompts = [prompts]

    model_instance.to(device).eval()

    prompts_out, completions_out = [], []

    for s in range(0, len(prompts), batch_size):
        batch_prompts_raw = prompts[s:s+batch_size]

        # render once per batch
        batch_rendered = [
            model_tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in batch_prompts_raw
        ]
        enc = model_tokenizer(
            batch_rendered, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        T_in = enc["input_ids"].size(1)

        # bucket completions per prompt index within this batch
        per_prompt_bucket = [[] for _ in range(len(batch_prompts_raw))]

        remaining = n_responses_total
        while remaining > 0:
            cur_n = min(responses_per_call, remaining)

            with torch.no_grad():
                out_ids = model_instance.generate(
                    **enc, num_return_sequences=cur_n, **gen_kwargs
                )  # [B*cur_n, T_in + gen_len]

            comp_ids = out_ids[:, T_in:]
            decoded = model_tokenizer.batch_decode(comp_ids, skip_special_tokens=True)

            # outputs are stacked: p0 x cur_n, p1 x cur_n, ...
            total = comp_ids.size(0)  # B*cur_n
            for k in range(total):
                base_i = k // cur_n  # which prompt in this subcall
                per_prompt_bucket[base_i].append(decoded[k])

            # free between subcalls
            del out_ids, comp_ids, decoded
            if device == 'cuda':
                torch.cuda.empty_cache()

            remaining -= cur_n

        # now FLATTEN in prompt order so each prompt's 1..N completions are contiguous
        for i, completions in enumerate(per_prompt_bucket):
            for c in completions:
                prompts_out.append(batch_prompts_raw[i])  # original chat object
                completions_out.append(c)

        del enc
        if device == 'cuda':
            torch.cuda.empty_cache()

    return Dataset.from_dict({"prompt": prompts_out, "completion": completions_out})

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
    tokenizer,                  # same tokenizer for both models
    target_model,
    rows,                       # dict | list[dict] | Dataset | IterableDataset | list[(prompt, completion)]
    device='cuda',
    batch_size=8,
    max_length=1024,
    response_only=True,
    add_generation_prompt=True,
):
    # ---- normalize input to a list of dicts {'prompt':..., 'completion':...} ----
    if isinstance(rows, dict):
        rows = [rows]
    elif isinstance(rows, Dataset):
        # avoids Python-level slicing loops; fast path provided by HF
        rows = rows.to_list()
    elif isinstance(rows, IterableDataset):
        rows = list(rows)  # materialize (only if iterable)
    elif isinstance(rows, (list, tuple)):
        pass
    else:
        raise TypeError(f"`rows` must be dict/list/Dataset/IterableDataset, got {type(rows).__name__}")

    ref_model.to(device).eval()
    target_model.to(device).eval()

    out_kl = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]

        # render prompts + take completions as-is
        prompts_rendered, completions = [], []
        for ex in batch:
            if isinstance(ex, dict) and "prompt" in ex and "completion" in ex:
                p, c = ex["prompt"], ex["completion"]
            elif isinstance(ex, (list, tuple)) and len(ex) == 2:
                p, c = ex[0], ex[1]
            else:
                raise ValueError("Each item must be {'prompt':..., 'completion':...} or (prompt, completion).")

            # chat -> text if needed
            if isinstance(p, list) and p and isinstance(p[0], dict) and "role" in p[0]:
                p_txt = tokenizer.apply_chat_template(
                    p, add_generation_prompt=add_generation_prompt, tokenize=False
                )
            else:
                p_txt = str(p)

            prompts_rendered.append(p_txt)
            completions.append(str(c))

        full_txt = [p + c for p, c in zip(prompts_rendered, completions)]

        # tokenize
        enc_full = tokenizer(full_txt, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        enc_prompt = tokenizer(prompts_rendered, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

        input_ids = enc_full["input_ids"]              # [B, T]
        attn_mask = enc_full["attention_mask"]         # [B, T]
        prompt_len = enc_prompt["attention_mask"].sum(dim=1).to(torch.long)  # [B]

        # teacher-forced forwards
        ref_logits    = ref_model(input_ids=input_ids,    attention_mask=attn_mask).logits   # [B, T, V]
        target_logits = target_model(input_ids=input_ids, attention_mask=attn_mask).logits   # [B, T, V]

        ref_logp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)    # [B, T-1, V]
        tgt_logp = F.log_softmax(target_logits[:, :-1, :], dim=-1) # [B, T-1, V]
        tgt_p    = tgt_logp.exp()
        mask     = attn_mask[:, 1:].to(ref_logp.dtype).clone()     # [B, T-1]

        if response_only:
            # after shift, prompt occupies indices [0 .. prompt_len-2]
            cuts = (prompt_len - 1).clamp_min(0)
            for i, cut in enumerate(cuts.tolist()):
                if cut > 0:
                    mask[i, :min(cut, mask.shape[1])] = 0.0

        per_tok_kl = (tgt_p * (tgt_logp - ref_logp)).sum(dim=-1)   # [B, T-1]
        kl_sum = (per_tok_kl * mask).sum(dim=-1)                   # [B]
        out_kl.append(kl_sum.detach().cpu())

        # cleanup
        del enc_full, enc_prompt, ref_logits, target_logits, ref_logp, tgt_logp, tgt_p, mask, per_tok_kl, kl_sum
        if device == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(out_kl).tolist()


@torch.no_grad()
def forward(model, model_tok, prompt, completion, temperature=1.0):
    
    if isinstance(prompts, str):
        prompt = [prompt]
    elif (isinstance(prompt, list) and prompts
          and isinstance(prompt[0], dict) and "role" in prompt[0]):
        prompt = [prompt]
        
    
    prompt = model_tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    enc_prompt = model_tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to('cuda')
    
    enc_completion = model_tok(
        completion,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to('cuda')
    
    model.to('cuda').eval()
    
    
    prompt_ids, prompt_attention_mask = enc_prompt['input_ids'], enc_prompt['attention_mask']
    completion_ids, completion_attention_mask = enc_completion['input_ids'], enc_completion['attention_mask']
    
    
    # print("_forward, prompt_ids.shape: ",prompt_ids.shape)
    # Concat the prompt and completion
    prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    # print("_forward, prompt_completion_ids.shape: ",prompt_completion_ids.shape)
    # Get the logps of the completions from the model
    output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
    # print("_forward, output.logits.shape: ",output.logits.shape)
    # There is 1 offset, because the model predict the next token
    logits = output.logits[:, max(0, prompt_ids.size(1) - 1) : -1]
    if temperature > 0:
        logits /= temperature + 1e-7

    if completion_ids.size(1) > logits.size(1):
        completion_ids = completion_ids[:, : logits.size(1)]

    # print("_forward, logits.shape: ",logits.shape)
    # Take the completion tokens logp
    logps = selective_log_softmax(logits, completion_ids)
    del (output, logits, prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask, prompt_completion_ids, prompt_completion_mask)
    # logps = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
    return logps.sum(-1)


def calculate_bias(ds, ref_model, ref_tok, dpo_model, n_responses_total = 10):
    
    prompts = ds['prompt']
    a1 = ds['a1']
    a2 = ds['a2']
    rank = ds['rank']
    bias = []
    
    for prompt in tqdm(prompts):
        
        new_ds = generate_responses(ref_model, ref_tok, prompt, max_new_tokens = 512, n_responses_total = n_responses_total)
        new_completions = new_ds['completion']
        kl_ref = get_expected_kl(ref_model, ref_tok, dpo_model, new_ds)
        mean_kl_ref = -0.1 * np.mean(kl_ref)
        r_ref = []
        for c in new_completions:
            r = 0.1 * (forward(dpo_instance, dpo_tok, prompt, c, temperature=1.0) - forward(ref_instance, ref_tok, prompt, c, temperature=1.0))
            r_ref.append(r.item())
        mean_r_ref = np.mean(r_ref)
        bias_prompt = mean_kl_ref - mean_r_ref
        bias.append(bias_prompt)
    
    return {
        'prompt':prompts,
        'a1':a1,
        'a2':a2,
        'rank':rank,
        'bias':bias
    }
    
    
def process_in_chunks_and_push(
    full_ds,
    *,
    ref_model,
    ref_tok,
    dpo_model,
    n_responses_total=10,
    chunk_size=10_000,
    map_batch_size=8,
    repo_base="your-username/bias-dataset",
    start_offset=0,   # resume from a given row if desired
):
    """
    Process `full_ds` in 10k-sized chunks, push a NEW repository after each chunk:
       repo_base-10000k  (first 10k rows)
       repo_base-20000k  (first 20k rows)
       ...
    Keeps a cumulative Dataset in memory (concatenated) so each push is cumulative.

    Args:
      full_ds: HF datasets Dataset with columns ['prompt','a1','a2','rank', ...]
      ref_model, ref_tok, dpo_model: your models/tokenizers used inside `calculate_bias`
      n_responses_total: passed to `calculate_bias`
      chunk_size: number of prompts per chunk (default 10_000)
      map_batch_size: batches inside datasets.map (default 8)
      repo_base: base repo id, e.g. "username/my-bias"
      private: whether to create the repos private
      hf_token: HF API token (if None, use cached login)
      start_offset: row index to resume from

    Returns:
      None (pushes to Hub after each chunk)
    """

    n = len(full_ds)
    if start_offset >= n:
        raise ValueError(f"start_offset={start_offset} >= dataset length {n}")

    cumulative = None
    processed = start_offset

    chunk_id = 0
    while processed < n:
        chunk_id += 1
        end = min(processed + chunk_size, n)
        print(f"\n=== Processing rows [{processed}:{end}) (size={end-processed}) ===")

        chunk = full_ds.select(range(processed, end))

        # Compute bias for this chunk (disable cache to force recompute)
        chunk_with_bias = chunk.map(
            calculate_bias,
            batched=True,
            batch_size=map_batch_size,
            fn_kwargs=dict(
                ref_model=ref_model,
                ref_tok=ref_tok,
                dpo_model=dpo_model,
                n_responses_total=n_responses_total,
            ),
            load_from_cache_file=False,
            desc=f"calculate_bias [{processed}:{end})"
        )

        # Build the cumulative dataset to date
        if cumulative is None:
            cumulative = chunk_with_bias
        else:
            cumulative = concatenate_datasets([cumulative, chunk_with_bias])

        # Determine repo name like "...-10000k", "...-20000k", etc.
        total_so_far = end  # cumulative rows
        suffix_k = f"{total_so_far//1000:05d}k"  # zero-padded thousands, e.g. 00010k
        repo_id = f"{repo_base}-{suffix_k}"

        # Push as a single-split Dataset; will create repo if missing
        cumulative.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Add rows 0..{end-1} (total {total_so_far})"
        )

        processed = end

    print("\nAll chunks processed and pushed.")
    

if __name__ == '__main__':
    
    
    ref_instance, ref_tok = load_model(ref_policy_path, task = 'generation')
    dpo_instance, dpo_tok = load_model(dpo_policy_path, task = 'generation')
    ds = load_dataset(ds_path, cache_dir=data_cache_path, split = 'train')
    prompts = ds['prompt']

    process_in_chunks_and_push(
        ds,
        ref_model=ref_instance,
        ref_tok=ref_tok,
        dpo_model=dpo_instance,
        n_responses_total=5,   # your current setting
        chunk_size=2000,      # push every 10k
        map_batch_size=8,
        repo_base="august66/hh_qwen2.5_1.5b_with_bias",
        start_offset=0
    )