import sys
import os
import csv

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trl import HfPairwiseJudge, BasePairwiseJudge, OpenAIPairwiseJudge
from datasets import load_dataset
import pandas as pd
import random
from functools import partial
from threading import Lock
from typing import Optional, Union
from concurrent.futures import TimeoutError
from huggingface_hub import InferenceClient
from openai import OpenAI
  

import json
import logging
import time
import tempfile
import os
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np

# def is_openai_available():
#     try:
#         OpenAI()
#         return True
#     except Exception:
#         return False


COMPARISON_PATH = """./gpt_evaluation/tldr/{method_name}_temp_{temp_str}_results_base_C.csv"""
SOURCE_DATA = "Eehan/eval-tldr"
SOURCE_LENGTH = 3000
SUMMARY_PATH = f"./gpt_evaluation/tldr/apocrypha_base_C.csv"
temperatures = [1, 0.75, 0.5]

DEFAULT_PAIRWISE_SYSTEM_PROMPT = """
Which of the following summaries does a better job of summarizing the most
important points in the given forum post?
Post: {user_query}
Summary A: {response_a}
Summary B: {response_b}
FIRST provide a one-short-sentence comparison of the two summaries, explaining which
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
choice. Your response should use the format:
Comparison: <one-sentence comparison and short explanation>
Preferred: <"A" or "B">
"""

# A simpler prompt for the judge method that expects "0" or "1"
DEFAULT_JUDGE_SYSTEM_PROMPT = """
Which of the following summaries does a better job of summarizing the most
important points in the given forum post?
Post: {user_query}
Summary A: {response_a}
Summary B: {response_b}
Respond with "A" or "B" to indicate your choice. Do not provide any other text.
"""

# system_prompt = """Which of the following summaries does a better job of summarizing the post? 
# Strictly follow these criteria when selecting the best summary: 
# 1. Prioritize the summary which eliminates unnecessary details and keeps only the author’s main concern or question. 
# 2. Avoid lengthy sentences, minor details or redundant information, express the key idea in few words. 
# 3. Prioritize the shorter summary as long as it remains clear and preserves the main idea.  
# Post: {prompt}. 
# Summary 0: {response0}, Summary 1: {response1}, 
# state only "0" or "1" to indicate your choice."""

system_prompt = """Which of the following summaries does a better job of 
summarizing the most important points in the given forum post, 
without including unimportant or irrelevant details? 
A good summary is both precise and concise.
Post: {user_query}
Summary A: {response_a}
Summary B: {response_b}
FIRST provide a one-sentence comparison of the two summaries, 
explaining which you prefer and why. 
SECOND, on a new line, state only "A" or "B" to indicate your choice. 
Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">"""

# system_prompt = """Which of the following summaries does a better job of summarizing the most
# important points in the given forum post?
# Post: {user_query}
# Summary A: {response_a}
# Summary B: {response_b}
# FIRST provide a one-short-sentence comparison of the two summaries, explaining which
# you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
# choice. Your response should use the format:
# Comparison: <one-sentence comparison and short explanation>
# Preferred: <"A" or "B">"""

# class OpenAIPairwiseJudge(BasePairwiseJudge):
#     """
#     Judge based on the OpenAI API, utilizing the Batch API for processing.

#     This judge is relevant for assessing the quality of chat models, where the completion is a response to a given prompt.

#     Args:
#         model (`str`, *optional*, defaults to `"gpt-4-turbo-preview"`):
#             Model to use for the judge.
#         system_prompt (`str` or `None`, *optional*, defaults to `None`):
#             System prompt to be used for the `judge_with_reason` method. If not provided, a default prompt is used.
#             It should contain placeholders: `{user_query}`, `{response_a}`, and `{response_b}`.
#         judge_system_prompt (`str` or `None`, *optional*, defaults to `None`):
#             System prompt for the `judge` method (expects "0" or "1"). If not provided, a default is used.
#             It should contain placeholders: `{user_query}`, `{response_0}`, and `{response_1}`.
#         max_requests (`int` or `None`, *optional*, defaults to `100000000`):
#             Maximum number of individual comparisons to make across all calls. If set to `None`, there is no limit.
#         batch_completion_window (`str`, *optional*, defaults to `"24h"`):
#             The time window for the batch job to complete.
#         poll_interval_seconds (`int`, *optional*, defaults to `30`):
#             How often to poll for batch job status.
#     """

#     def __init__(
#         self,
#         model: str = "gpt-4-turbo-preview",
#         system_prompt: Optional[str] = None,
#         judge_system_prompt: Optional[str] = None,
#         max_requests: Union[int, None] = 100000000,
#         batch_completion_window: str = "24h",
#         poll_interval_seconds: int = 30,
#     ):
#         if not is_openai_available():
#             raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
#         try:
#             self.client = OpenAI()
#             # Test client connection (optional, but good for early failure)
#             # Changed: Removed 'limit=1' as it caused TypeError with some openai library versions.
#             # A successful call to list() is enough to test basic connectivity and authentication.
#             self.client.models.list() 
#         except Exception as e:
#             # Catch the specific error from the traceback if possible, or a general one.
#             if "got an unexpected keyword argument 'limit'" in str(e) and "self.client.models.list(limit=1)" in str(e.__traceback__):
#                  # This specific error path might be less likely now with the fix,
#                  # but keeping the structure in case other init errors occur.
#                  pass # The error is now related to the modified call if it still fails.
#             raise ValueError(f"Failed to initialize OpenAI client. Please check API key and connectivity. Error: {e}")

#         self.model = model
#         self.system_prompt_for_reason = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT
#         self.system_prompt_for_judge = judge_system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
#         self.max_requests = max_requests
#         self.batch_completion_window = batch_completion_window
#         self.poll_interval_seconds = poll_interval_seconds
        
#         self.num_requests = 0
#         self._warned_max_requests = False

#     def _prepare_batch_requests(
#         self,
#         prompts: List[str],
#         completions: List[List[str]],
#         is_judge_with_reason: bool,
#     ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
#         """Prepares the data for a batch API call."""
#         batch_requests_data = []
#         custom_id_to_index_map: Dict[str, int] = {}

#         for i, (prompt_text, candidate_pair) in enumerate(zip(prompts, completions)):
#             if self.max_requests is not None and (self.num_requests + len(batch_requests_data)) >= self.max_requests:
#                 logging.warning(
#                     f"Reached effective max_requests limit ({self.max_requests}) during batch preparation. "
#                     f"Only processing {len(batch_requests_data)} items for this batch."
#                 )
#                 break # Stop adding more requests to this batch

#             custom_id = f"judge_item_{i}_{time.time_ns()}" # Ensure unique custom_id
#             custom_id_to_index_map[custom_id] = i

#             if is_judge_with_reason:
#                 content = self.system_prompt_for_reason.format(
#                     user_query=prompt_text, response_a=candidate_pair[0], response_b=candidate_pair[1]
#                 )
#                 max_tokens = 200 # For detailed reason
#             else: # For judge method (0 or 1)
#                 content = self.system_prompt_for_judge.format(
#                     user_query=prompt_text, response_0=candidate_pair[0], response_1=candidate_pair[1]
#                 )
#                 max_tokens = 5 # Allow a bit more for robustness, expecting "0" or "1"

#             messages = [{"role": "user", "content": content}]
            
#             batch_requests_data.append({
#                 "custom_id": custom_id,
#                 "method": "POST",
#                 "url": "/v1/chat/completions", # Standard endpoint for chat models
#                 "body": {
#                     "model": self.model,
#                     "messages": messages,
#                     "max_tokens": max_tokens,
#                     # "temperature": 0, # Optional: for more deterministic output
#                 },
#             })
#         return batch_requests_data, custom_id_to_index_map

#     def _submit_and_process_batch(
#         self,
#         batch_requests_data: List[Dict[str, Any]],
#         num_original_prompts: int # This argument seems unused, consider removing if not needed.
#     ) -> Dict[str, Any]:
#         """
#         Submits requests as a batch job and waits for results.
#         Returns a dictionary mapping custom_id to the API response content or an error.
#         """
#         if not batch_requests_data:
#             return {}

#         results_map: Dict[str, Any] = {}
        
#         # Create a temporary file for batch input
#         # Ensure the temporary file is properly handled for deletion
#         tmp_file_path = "" # Initialize to ensure it's defined in finally
#         try:
#             with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp_file:
#                 for req_data in batch_requests_data:
#                     tmp_file.write(json.dumps(req_data) + "\n")
#                 tmp_file_path = tmp_file.name # Store path for later deletion
            
#             # 1. Upload file
#             logging.info(f"Uploading batch input file: {tmp_file_path}")
#             with open(tmp_file_path, "rb") as f:
#                 batch_input_file = self.client.files.create(file=f, purpose="batch")
#             logging.info(f"Batch input file uploaded with ID: {batch_input_file.id}")

#             # 2. Create batch job
#             logging.info(f"Creating batch job for file ID: {batch_input_file.id}")
#             created_batch = self.client.batches.create(
#                 input_file_id=batch_input_file.id,
#                 endpoint="/v1/chat/completions", # This must match the 'url' in your JSONL
#                 completion_window=self.batch_completion_window,
#                 metadata={"description": f"Pairwise judging batch with {len(batch_requests_data)} items"},
#             )
#             batch_id = created_batch.id
#             logging.info(f"Batch job created with ID: {batch_id}, Status: {created_batch.status}")

#             # 3. Poll for completion
#             while True:
#                 batch_job = self.client.batches.retrieve(batch_id)
#                 logging.info(f"Polling batch job {batch_id}. Status: {batch_job.status}, "
#                              f"Requests: {batch_job.request_counts.completed if batch_job.request_counts else 'N/A'}/{batch_job.request_counts.total if batch_job.request_counts else 'N/A'} done, "
#                              f"{batch_job.request_counts.failed if batch_job.request_counts else 'N/A'} failed.")


#                 if batch_job.status == "completed":
#                     logging.info(f"Batch job {batch_id} completed.")
#                     if batch_job.output_file_id:
#                         output_content_response = self.client.files.content(batch_job.output_file_id)
#                         output_data = output_content_response.read().decode()
#                         for line in output_data.strip().split('\n'):
#                             if not line: continue
#                             item_result = json.loads(line)
#                             custom_id = item_result.get("custom_id")
#                             if custom_id:
#                                 # Store the 'body' of the successful response
#                                 if item_result.get("response") and item_result["response"].get("body"):
#                                      results_map[custom_id] = item_result["response"]["body"]
#                                 elif item_result.get("error"): # Check for per-item error in output
#                                     results_map[custom_id] = {"error": item_result["error"]}
#                                 else: # Fallback if structure is unexpected
#                                     results_map[custom_id] = {"error": {"message": "Unknown structure in output item."}}
#                     else:
#                         logging.warning(f"Batch job {batch_id} completed but no output_file_id was present.")
#                         # If no output file, it might mean all requests failed and are in error_file_id,
#                         # or there were no successful requests. We should still mark them as errored if not already.
#                         for req_data in batch_requests_data:
#                             custom_id = req_data["custom_id"]
#                             if custom_id not in results_map:
#                                 results_map[custom_id] = {"error": {"message": "Completed with no output file, presumed error."}}


#                     if batch_job.error_file_id:
#                         logging.warning(f"Batch job {batch_id} has an error file: {batch_job.error_file_id}")
#                         # Optionally download and process error file for more detailed per-item errors
#                         try:
#                             error_content_response = self.client.files.content(batch_job.error_file_id)
#                             error_data = error_content_response.read().decode()
#                             for line in error_data.strip().split('\n'):
#                                 if not line: continue
#                                 error_item = json.loads(line)
#                                 custom_id = error_item.get("custom_id")
#                                 if custom_id and custom_id not in results_map: # Only add if not already processed from main output
#                                     results_map[custom_id] = {"error": error_item.get("response", {}).get("body", {"message": "Error detail from error file."})}
#                         except Exception as e_err_file:
#                             logging.error(f"Failed to process error file {batch_job.error_file_id}: {e_err_file}")
#                     break
#                 elif batch_job.status in ["failed", "cancelled", "expired"]:
#                     logging.error(f"Batch job {batch_id} ended with status: {batch_job.status}. Errors: {batch_job.errors}")
#                     # Populate all pending requests with an error
#                     for req_data in batch_requests_data:
#                         custom_id = req_data["custom_id"]
#                         if custom_id not in results_map: # Avoid overwriting if already processed
#                              results_map[custom_id] = {"error": {"message": f"Batch job status: {batch_job.status}"}}
#                     break
                
#                 time.sleep(self.poll_interval_seconds)
        
#         except openai.APIError as e:
#             logging.error(f"OpenAI API error during batch processing: {e}")
#             # Populate all pending requests with this general error
#             for req_data in batch_requests_data:
#                 custom_id = req_data["custom_id"]
#                 if custom_id not in results_map:
#                     results_map[custom_id] = {"error": {"message": f"Batch API Error: {str(e)}"}}
#         finally:
#             if tmp_file_path and os.path.exists(tmp_file_path):
#                 try:
#                     os.remove(tmp_file_path)
#                     logging.info(f"Temporary batch input file {tmp_file_path} removed.")
#                 except Exception as e_remove:
#                     logging.error(f"Error removing temporary file {tmp_file_path}: {e_remove}")
        
#         return results_map

#     def judge_with_reason(
#         self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True
#     ) -> Tuple[List[int], List[str]]:
#         if not prompts: # Handle empty input
#             return [], []
            
#         if self.max_requests is not None and self.num_requests >= self.max_requests:
#             if not self._warned_max_requests:
#                 logging.warning(
#                     f"Reached the maximum number of requests ({self.max_requests}). Returning default error values."
#                     " Set `max_requests` to None for no limit."
#                 )
#                 self._warned_max_requests = True
#             return [-1] * len(prompts), ["Max requests reached"] * len(prompts)

#         # Determine how many items can be processed based on max_requests
#         effective_len_prompts = len(prompts)
#         num_can_process = effective_len_prompts
#         if self.max_requests is not None:
#             remaining_requests_allowance = self.max_requests - self.num_requests
#             if remaining_requests_allowance <= 0:
#                 if not self._warned_max_requests: # Warning for this specific call
#                     logging.warning(f"Max requests ({self.max_requests}) already met. No items from this call will be processed.")
#                     self._warned_max_requests = True # Ensure global warning is set
#                 return [-1] * effective_len_prompts, ["Max requests reached before call"] * effective_len_prompts
            
#             num_can_process = min(effective_len_prompts, remaining_requests_allowance)
#             if num_can_process < effective_len_prompts:
#                  logging.warning(f"Processing only {num_can_process} of {effective_len_prompts} prompts due to max_requests limit.")
        
#         # Slice inputs if only a subset can be processed
#         prompts_to_process = prompts[:num_can_process]
#         completions_to_process = completions[:num_can_process]
        
#         if not prompts_to_process: # If slicing results in no prompts to process
#             # This case might occur if num_can_process is 0 but original prompts list was not empty.
#             # Return appropriately sized error lists for the original number of prompts.
#             return [-1] * effective_len_prompts, ["Not processed due to max_requests limit"] * effective_len_prompts


#         flip_mask = np.array([False] * num_can_process)
#         if shuffle_order:
#             flip_mask = np.random.choice([True, False], size=num_can_process)
#             shuffled_completions = [
#                 pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions_to_process)
#             ]
#         else:
#             shuffled_completions = completions_to_process

#         batch_requests_data, custom_id_to_index_map = self._prepare_batch_requests(
#             prompts_to_process, shuffled_completions, is_judge_with_reason=True
#         )
        
#         if not batch_requests_data: 
#             # This means _prepare_batch_requests decided not to create any items,
#             # possibly due to an internal check against self.num_requests again.
#             # Return error values for the number of items we intended to process.
#             return [-1] * num_can_process + ([-1] * (effective_len_prompts - num_can_process)), \
#                    ["Batch preparation failed or limit hit"] * num_can_process + \
#                    (["Not processed due to max_requests limit"] * (effective_len_prompts - num_can_process))


#         batch_results_map = self._submit_and_process_batch(batch_requests_data, num_can_process)

#         # Initialize results for the number of items we attempted to process
#         ranks_processed = [-1] * num_can_process
#         reasons_processed = ["Error or no response"] * num_can_process

#         for custom_id, original_idx in custom_id_to_index_map.items():
#             # original_idx is an index within the prompts_to_process list
#             if original_idx >= num_can_process: continue 

#             api_response_data = batch_results_map.get(custom_id)
#             if api_response_data and "error" not in api_response_data:
#                 try:
#                     response_content = api_response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
#                     lines = [line.strip() for line in response_content.split('\n')]
#                     comparison = next((line for line in lines if line.lower().startswith("comparison:")), None)
#                     helpful_line = next((line for line in reversed(lines) if line.lower().startswith("preferred:")), None)
                    
#                     reason_text = comparison.split(":", 1)[1].strip() if comparison else "No comparison provided"
#                     choice_text = helpful_line.split(":")[-1].strip().upper() if helpful_line else None

#                     if choice_text == "A":
#                         ranks_processed[original_idx] = 0
#                         reasons_processed[original_idx] = reason_text
#                     elif choice_text == "B":
#                         ranks_processed[original_idx] = 1
#                         reasons_processed[original_idx] = reason_text
#                     else:
#                         ranks_processed[original_idx] = -1 
#                         reasons_processed[original_idx] = f"Invalid choice: {choice_text}. Raw: {response_content[:100]}"
#                 except Exception as e:
#                     ranks_processed[original_idx] = -1
#                     reasons_processed[original_idx] = f"Error parsing response: {str(e)}. Raw: {str(api_response_data)[:200]}" # Increased raw output
#             elif api_response_data and "error" in api_response_data:
#                 ranks_processed[original_idx] = -1
#                 reasons_processed[original_idx] = f"API Error: {api_response_data['error'].get('message', 'Unknown error')}"
#             else: 
#                 ranks_processed[original_idx] = -1
#                 reasons_processed[original_idx] = "No result found for item in batch."


#         if shuffle_order:
#             final_ranks_processed = [
#                 (1 - rank) if rank != -1 and flip else rank
#                 for rank, flip in zip(ranks_processed, flip_mask) # flip_mask is already sized for num_can_process
#             ]
#         else:
#             final_ranks_processed = ranks_processed
        
#         self.num_requests += len(batch_requests_data) 
        
#         # Combine processed results with placeholders for unprocessed items
#         final_ranks_all = final_ranks_processed + [-1] * (effective_len_prompts - num_can_process)
#         final_reasons_all = reasons_processed + ["Not processed due to max_requests limit"] * (effective_len_prompts - num_can_process)
        
#         return final_ranks_all, final_reasons_all

#     def judge(
#         self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True
#     ) -> List[int]:
#         if not prompts: # Handle empty input
#             return []

#         if self.max_requests is not None and self.num_requests >= self.max_requests:
#             if not self._warned_max_requests:
#                 logging.warning(
#                     f"Reached the maximum number of requests ({self.max_requests}). Returning -1."
#                     " Set `max_requests` to None for no limit."
#                 )
#                 self._warned_max_requests = True
#             return [-1] * len(prompts)
        
#         effective_len_prompts = len(prompts)
#         num_can_process = effective_len_prompts
#         if self.max_requests is not None:
#             remaining_requests_allowance = self.max_requests - self.num_requests
#             if remaining_requests_allowance <= 0:
#                 if not self._warned_max_requests:
#                      logging.warning(f"Max requests ({self.max_requests}) already met. No items from this call will be processed.")
#                      self._warned_max_requests = True
#                 return [-1] * effective_len_prompts
            
#             num_can_process = min(effective_len_prompts, remaining_requests_allowance)
#             if num_can_process < effective_len_prompts:
#                  logging.warning(f"Processing only {num_can_process} of {effective_len_prompts} prompts due to max_requests limit.")

#         prompts_to_process = prompts[:num_can_process]
#         completions_to_process = completions[:num_can_process]

#         if not prompts_to_process:
#             return [-1] * effective_len_prompts
            
#         flip_mask = np.array([False] * num_can_process)
#         if shuffle_order:
#             flip_mask = np.random.choice([True, False], size=num_can_process)
#             shuffled_completions = [
#                 pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions_to_process)
#             ]
#         else:
#             shuffled_completions = completions_to_process
        
#         batch_requests_data, custom_id_to_index_map = self._prepare_batch_requests(
#             prompts_to_process, shuffled_completions, is_judge_with_reason=False
#         )

#         if not batch_requests_data:
#             return [-1] * num_can_process + ([-1] * (effective_len_prompts - num_can_process))


#         batch_results_map = self._submit_and_process_batch(batch_requests_data, num_can_process)
        
#         ranks_processed = [-1] * num_can_process

#         for custom_id, original_idx in custom_id_to_index_map.items():
#             if original_idx >= num_can_process: continue

#             api_response_data = batch_results_map.get(custom_id)
#             if api_response_data and "error" not in api_response_data:
#                 try:
#                     response_content = api_response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
#                     if response_content in ["0", "1"]:
#                         ranks_processed[original_idx] = int(response_content)
#                     else:
#                         logging.debug(f"Invalid response for judge: '{response_content}'. Expected '0' or '1'. Custom ID: {custom_id}")
#                         ranks_processed[original_idx] = -1 
#                 except Exception as e:
#                     logging.error(f"Error parsing judge response: {e}. Raw: {str(api_response_data)[:200]}. Custom ID: {custom_id}")
#                     ranks_processed[original_idx] = -1
#             elif api_response_data and "error" in api_response_data:
#                 ranks_processed[original_idx] = -1
#                 logging.warning(f"API Error for custom_id {custom_id}: {api_response_data['error'].get('message', 'Unknown error')}")
#             else:
#                 ranks_processed[original_idx] = -1
#                 logging.warning(f"No result found for custom_id {custom_id} in batch.")


#         if shuffle_order:
#             final_ranks_processed = [
#                 (1 - rank) if rank != -1 and flip else rank
#                 for rank, flip in zip(ranks_processed, flip_mask)
#             ]
#         else:
#             final_ranks_processed = ranks_processed
            
#         self.num_requests += len(batch_requests_data)

#         final_ranks_all = final_ranks_processed + [-1] * (effective_len_prompts - num_can_process)
            
#         return final_ranks_all



def evaluate_and_save(data, method_name, temp):
        results, reasons = judge.judge_with_reason(
            prompts=data["prompt"],
            completions=data["completions"]
        )
        
        df = pd.DataFrame({
            "prompt": data["prompt"],
            "completion_1": [c[0] for c in data["completions"]],
            "completion_2": [c[1] for c in data["completions"]],
            "judge_result": results,
            "judge_reason": reasons,
        })
        
        win_rate = sum(results) / len(results)
        print(f"{method_name}, {temp}: {win_rate}")
        
        temp_str = str(temp).replace(".", "")
        df.to_csv(COMPARISON_PATH.format(method_name=method_name, temp_str=temp_str), index=False, escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
        
        return win_rate


# def evaluate_and_save(data, method_name, temp_value):
#     """
#     Evaluates data using the judge and saves results.
#     The judge_instance is passed to allow using the same instance.
#     """
#     if not data["prompt"]: # Handle empty data case
#         logging.warning(f"No prompts to evaluate for {method_name}, {temp_value}. Skipping.")
#         return 0.0 # Or handle as appropriate

#     logging.info(f"Starting evaluation for {method_name}, temperature {temp_value}...")
#     results, reasons = judge.judge_with_reason(
#         prompts=data["prompt"],
#         completions=data["completions"]
#     )
    
#     # Ensure results and reasons have the same length as prompts for DataFrame creation
#     # This handles cases where max_requests might truncate processing within the judge
#     num_evaluated = len(results)
    
#     df_prompts = data["prompt"][:num_evaluated]
#     df_completions1 = [c[0] for c in data["completions"][:num_evaluated]]
#     df_completions2 = [c[1] for c in data["completions"][:num_evaluated]]

#     # If fewer results were returned than prompts due to internal limits, pad the rest
#     if num_evaluated < len(data["prompt"]):
#         padding_count = len(data["prompt"]) - num_evaluated
#         df_prompts.extend(data["prompt"][num_evaluated:])
#         df_completions1.extend(["N/A"] * padding_count) # Or original data if preferred
#         df_completions2.extend(["N/A"] * padding_count)
#         results.extend([-1] * padding_count)
#         reasons.extend(["Not processed by judge"] * padding_count)


#     df = pd.DataFrame({
#         "prompt": df_prompts,
#         "completion_1": df_completions1,
#         "completion_2": df_completions2,
#         "judge_result": results,
#         "judge_reason": reasons, # Corrected typo from "judege_reason"
#     })
    
#     # Refined win rate calculation:
#     # Assumes '1' means the second completion (index 1) is preferred (a "win" for the second model in the pair).
#     # Excludes errors (where result is -1).
#     valid_results = [r for r in results if r != -1]
#     if not valid_results:
#         win_rate = 0.0
#         logging.warning(f"No valid results for {method_name}, {temp_value} to calculate win rate.")
#     else:
#         # Example: Win rate of the second completion (completion_2)
#         wins_for_completion_2 = sum(1 for r in valid_results if r == 1)
#         win_rate = wins_for_completion_2 / len(valid_results)
#         # If 0 means the first completion is preferred, and you want its win rate:
#         # wins_for_completion_1 = sum(1 for r in valid_results if r == 0)
#         # win_rate = wins_for_completion_1 / len(valid_results) 
#         # Adjust the definition of "win" as per your needs.

#     logging.info(f"Evaluation for {method_name}, temperature {temp_value}: Win rate = {win_rate:.4f} (based on {len(valid_results)} valid judgments)")
    
#     temp_str = str(temp_value).replace(".", "")
#     output_path = COMPARISON_PATH.format(method_name=method_name, temp_str=temp_str)
#     df.to_csv(output_path, index=False)
#     logging.info(f"Results saved to {output_path}")
    
#     return win_rate

if __name__ == "__main__":

    judge = OpenAIPairwiseJudge(model = "gpt-4o-mini", system_prompt=system_prompt)


    data = load_dataset(SOURCE_DATA)
    # print(data["temperature_0"].select(range(5)).to_pandas())
    all_indices = list(range(SOURCE_LENGTH))
    random.seed(66004)
    random.shuffle(all_indices)  
    num_samples_per_temp = SOURCE_LENGTH // len(data.keys())
    print(num_samples_per_temp)

    temp_indices = {
        temp: all_indices[i * num_samples_per_temp: (i + 1) * num_samples_per_temp]
        for i, temp in enumerate(temperatures)
    }
    temperature_data = {
        temp: [data[f"temperature_{temp}"][i] for i in temp_indices[temp]]
        for temp in temp_indices
    }

    results_df = pd.DataFrame(columns=["Temperature", "Model", "Win Rate"])
    for temp in temperatures:
        temp_data = temperature_data[temp]
        
        
        # data_sft_drpo_gpm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
        # data_dpo_drpo_gpm =  {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
        # data_sft_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["drpo-0066004-new"]] for x in temp_data]}
        # data_dpo_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["drpo-0066004-new"]] for x in temp_data]}
        # data_ipo_drpo_bt = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ipo"], x["drpo-0066004-new"]] for x in temp_data]}
        # data_ipo_drpo_gpm = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ipo"], x["drpo-gpm-2dim-0066004-new"]] for x in temp_data]}
        # data_sft_dpo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["dpo"]] for x in temp_data]}
        # data_sft_ppo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["ppo"]] for x in temp_data]}
        # # data_sft_ipo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["sft"], x["ipo"]] for x in temp_data]}
        # # data_dpo_ipo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["ipo"]] for x in temp_data]}
        # # data_ppo_ipo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["ipo"]] for x in temp_data]}
        # data_dpo_ppo = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["dpo"], x["ppo"]] for x in temp_data]}
        data_ppo_drpo88_1e = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo-1e-088004"]] for x in temp_data]}
        data_ppo_drpo88_07e = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo-0.7e-088004"]] for x in temp_data]}
        data_ppo_drpo88_gpm_1e = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo-gpm-1e-088004"]] for x in temp_data]}
        data_ppo_drpo88_gpm_09e = {"prompt": [x["prompt"] for x in temp_data], "completions": [[x["ppo"], x["drpo-gpm-0.9e-088004"]] for x in temp_data]}
                            
        
        # sft_drpo_gpm_win_rate = evaluate_and_save(data_sft_drpo_gpm, "sft_drpo-gpm", temp)
        # dpo_drpo_gpm_win_rate = evaluate_and_save(data_dpo_drpo_gpm, "dpo-drpo_gpm", temp)
        # sft_drpo_bt_win_rate = evaluate_and_save(data_sft_drpo_bt, "sft-drpo_bt", temp)
        # dpo_drpo_bt_win_rate = evaluate_and_save(data_dpo_drpo_bt, "dpo-drpo_bt", temp)
        # ipo_drpo_bt_win_rate = evaluate_and_save(data_ipo_drpo_bt, "ipo-drpo_bt", temp)
        # ipo_drpo_gpm_win_rate = evaluate_and_save(data_ipo_drpo_gpm, "ipo-drpo_gpm", temp)
        # sft_dpo_win_rate = evaluate_and_save(data_sft_dpo, "sft-dpo", temp)
        # sft_ppo_win_rate = evaluate_and_save(data_sft_ppo, "sft-ppo", temp)
        # # sft_ipo_win_rate = evaluate_and_save(data_sft_ipo, "sft-ipo", temp)
        # # dpo_ipo_win_rate = evaluate_and_save(data_dpo_ipo, "dpo-ipo", temp)
        # # ppo_ipo_win_rate = evaluate_and_save(data_ppo_ipo, "ppo-ipo", temp)
        # dpo_ppo_win_rate = evaluate_and_save(data_dpo_ppo, "dpo-ppo", temp)
        ppo_drpo88_1e_win_rate = evaluate_and_save(data_ppo_drpo88_1e, "ppo-drpo-1e", temp)
        ppo_drpo88_07e_win_rate = evaluate_and_save(data_ppo_drpo88_07e, "ppo-drpo-0.7e", temp)
        ppo_drpo88_gpm_1e_win_rate = evaluate_and_save(data_ppo_drpo88_gpm_1e, "ppo-drpo-gpm-1e", temp)
        ppo_drpo88_gpm_09e_win_rate = evaluate_and_save(data_ppo_drpo88_gpm_09e, "ppo-drpo-gpm-0.9e", temp)


        temp_results = [
            # {"Temperature": temp, "Model": "SFT_DRPO-GPM", "Win Rate": sft_drpo_gpm_win_rate},
            # {"Temperature": temp, "Model": "DPO_DRPO-GPM", "Win Rate": dpo_drpo_gpm_win_rate},
            # {"Temperature": temp, "Model": "SFT_DRPO-BT", "Win Rate": sft_drpo_bt_win_rate},
            # {"Temperature": temp, "Model": "DPO_DRPO-BT", "Win Rate": dpo_drpo_bt_win_rate},
            # {"Temperature": temp, "Model": "IPO_DRPO-BT", "Win Rate": ipo_drpo_bt_win_rate},
            # {"Temperature": temp, "Model": "IPO_DRPO-GPM", "Win Rate": ipo_drpo_gpm_win_rate},
            # {"Temperature": temp, "Model": "SFT-DPO", "Win Rate": sft_dpo_win_rate},
            # {"Temperature": temp, "Model": "SFT-PPO", "Win Rate": sft_ppo_win_rate},
            # # {"Temperature": temp, "Model": "SFT-IPO", "Win Rate": sft_ipo_win_rate},
            # # {"Temperature": temp, "Model": "DPO-IPO", "Win Rate": dpo_ipo_win_rate},
            # # {"Temperature": temp, "Model": "PPO-IPO", "Win Rate": ppo_ipo_win_rate},
            # {"Temperature": temp, "Model": "DPO-PPO", "Win Rate": dpo_ppo_win_rate},
            {"Temperature": temp, "Model": "PPO-DRPO-1e", "Win Rate": ppo_drpo88_1e_win_rate},
            {"Temperature": temp, "Model": "PPO-DRPO-0.7e", "Win Rate": ppo_drpo88_07e_win_rate},
            {"Temperature": temp, "Model": "PPO-DRPO-GPM-1e", "Win Rate": ppo_drpo88_gpm_1e_win_rate},
            {"Temperature": temp, "Model": "PPO-DRPO-GPM-0.9e", "Win Rate": ppo_drpo88_gpm_09e_win_rate},
        ]
        
        results_df = pd.concat([results_df, pd.DataFrame(temp_results)], ignore_index=True)


    results_df.to_csv(SUMMARY_PATH, index=False)
    # results_df.to_csv(SUMMARY_PATH, mode='a', header=False, index=False)
    print("Clear by Spring")