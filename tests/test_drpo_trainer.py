# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from unittest.mock import MagicMock
import os

import numpy as np
import torch
from datasets import Dataset, features, load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    is_vision_available,
)
from transformers.testing_utils import require_peft, require_torch_gpu_if_bnb_not_multi_backend_enabled, require_vision

from trl import DPOConfig, DPOTrainer, FDivergenceType

from .testing_utils import require_bitsandbytes, require_no_wandb


if is_vision_available():
    from PIL import Image


class TestTokenizeRow(unittest.TestCase):
    def setUp(self):
        # Set up the mock tokenizer with specific behaviors
        self.tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 2

        # Define mock return values for the tokenizer's 'input_ids' for the different text inputs
        self.tokenizer.return_value = {
            "input_ids": {"The sky is": [464, 6766, 318], " blue": [4171], " green": [4077]}
        }

        # Define tokenizer behavior when called
        def mock_tokenizer_call(text, add_special_tokens):
            token_map = {
                "The sky is": {"input_ids": [464, 6766, 318]},
                " blue": {"input_ids": [4171]},
                " green": {"input_ids": [4077]},

            }
            return token_map[text]

        self.tokenizer.side_effect = mock_tokenizer_call

    def test_tokenize_row_no_truncation_no_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with no truncation and no special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=False,
            
        )

        # Assert the correct output without truncation or special tokens
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [464, 6766, 318],
                "chosen_input_ids": [4171, 2],  # eos_token added
                "rejected_input_ids": [4077, 2],  # eos_token added
            },
        )

    def test_tokenize_row_with_truncation(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with truncation
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=2,
            max_completion_length=1,
            add_special_tokens=False,
        )

        # Assert the correct output with truncation applied
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [6766, 318],  # truncated to the last 2 tokens
                "chosen_input_ids": [4171],  # truncated to 1 token
                "rejected_input_ids": [4077],  # truncated to 1 token
            },
        )

    def test_tokenize_row_with_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=True,
        )

        # Assert the correct output with special tokens added
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [0, 464, 6766, 318, 2],  # bos_token and eos_token added
                "chosen_input_ids": [4171, 2],  # eos_token added
                "rejected_input_ids": [4077, 2],  # eos_token added
            },
        )

    def test_tokenize_row_with_truncation_and_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with both truncation and special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=4,
            max_completion_length=1,
            add_special_tokens=True,
        )

        # Assert the correct output with both truncation and special tokens
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [464, 6766, 318, 2],  # truncated to 4 tokens with bos_token and eos_token
                "chosen_input_ids": [4171],  # truncated to 1 token
                "rejected_input_ids": [4077],  # truncated to 1 token
            },
        )


if __name__ == "__main__":
    unittest.main()
    ds = load_dataset("Kyleyee/tldr_test_tiny_data_drpo")
