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

from ..trl import DRPOConfig, DRPOTrainer, FDivergenceType

from .testing_utils import require_bitsandbytes, require_no_wandb

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set up the models and tokenizer using the test model
        self.model_id = "Kyleyee/Qwen2-0.5B-stf-hh"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load dataset
        raw_dataset = load_dataset("Kyleyee/tldr_test_tiny_data_drpo", "standard_prompt_only")
        self.raw_dataset = raw_dataset.map(lambda x: self.tokenizer(x["prompt"]), remove_columns=["prompt"])

    def test_basic_training(self):
        """Test basic DRPO training configuration and verify model updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Capture initial weights
            initial_model_weights = {}
            for name, param in self.model.named_parameters():
                initial_model_weights[name] = param.clone().detach()

            training_args = DRPOConfig(
                output_dir=tmp_dir,
                max_length=64,
                temperature=0,
                beta=0.1,
                dataset_num_proc=1,
                num_star = 1,               
            )

            # Create the trainer
            trainer = DRPOTrainer(
                model=self.model,
                
            )

            # Train the model
            trainer.train()

            # Check that the model weights have been updated
            for name, param in self.model.named_parameters():
                self.assertFalse(torch.equal(param, initial_model_weights[name]))


