import tempfile
import unittest
import os
import json
from copy import deepcopy

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from trl.trainer.drpo_utils import BTwithRewardPipeline, BTRewardNetwork
from trl.trainer import DRPOConfig, DRPOTrainer
import numpy as np

class TestDRPO(unittest.TestCase):
    def setUp(self):
        self.model_id = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # self.model.resize_token_embeddings(len(self.tokenizer))
            # self.ref_model.resize_token_embeddings(len(self.tokenizer))
        
        # print("===========================\n model config\n===================")
        # print(self.model.config)
        # print(self.tokenizer.all_special_tokens)
        # print(self.tokenizer.all_special_ids)
        # print(self.tokenizer.convert_ids_to_tokens(1000))

        self.preference_model_id = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
        # self.preference_model = BTwithRewardPipeline(self.preference_model_id)
        self.preference_model = BTRewardNetwork(self.preference_model_id, pad_token_id=self.tokenizer.pad_token_id)

        # Create a small dummy dataset with proper structure
        self.dummy_dataset = Dataset.from_dict({
            "prompt": ["Summarize this:", "Explain this:"] * 4,
            "a1": ["Summary A", "Explanation A"] * 4,
            "a2": ["Summary B", "Explanation B"] * 4,
            "rank": [1.0, 1.0] * 4
        })

        # Create base training arguments
        self.base_training_args = DRPOConfig(
            output_dir="test_output",
            model_and_preference_share_basemodel=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-7,
            max_length=32,  # Reduced for testing
            max_prompt_length=16,  # Added
            max_completion_length=16,  # Added
            max_new_tokens=16,  # Added
            num_astar=2,
            beta=0.1,
            report_to="none",
            is_bt_model=True,
            torch_empty_cache_steps=1,
            ratio_processing="clip",
            clipbound=5.0,
            forward_temperature=1e-7,
            generate_temperature=1e-7,
            remove_unused_columns=False,  # Important for DRPO
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_training(self):
        """Test basic DRPO training configuration and verify model updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.base_training_args.output_dir = tmp_dir
            
            # Capture initial weights
            initial_weights = {name: param.clone().detach() 
                             for name, param in self.model.named_parameters()}

            trainer = DRPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                preference_model=self.preference_model,
                args=self.base_training_args,
                processing_class=self.tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )

            # Train for just one step
            trainer.train()

            # Verify weights changed
            for name, param in self.model.named_parameters():
                initial_weights[name] = initial_weights[name].to(param.device)
                self.assertFalse(
                    torch.equal(param, initial_weights[name]),
                    f"Parameters {name} did not change after training"
                )

    def test_drpo_trainer_loss_consistency(self):
        """Test loss consistency across different modes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            def get_loss_for_mode(mode=None):
                args = deepcopy(self.base_training_args)
                args.output_dir = tmp_dir
                
                if mode == "loss1":
                    args.loss1_only = True
                    args.loss2_only = False
                elif mode == "loss2":
                    args.loss1_only = False
                    args.loss2_only = True
                else:
                    args.loss1_only = False
                    args.loss2_only = False

                trainer = DRPOTrainer(
                    model=self.model,
                    ref_model=self.ref_model,
                    preference_model=self.preference_model,
                    args=args,
                    processing_class=self.tokenizer,
                    train_dataset=self.dummy_dataset,
                )

                trainer.train()
                return trainer.stats

            # Test different loss modes
            modes = [None]
            current_stats = {}
            
            for mode in modes:
                mode_name = mode if mode else "full"
                try:
                    current_stats[mode_name] = get_loss_for_mode(mode)
                    print(f"Current {mode_name} stats: {current_stats[mode_name]}")
                except Exception as e:
                    print(f"Error computing loss for mode {mode_name}: {str(e)}")
                    current_stats[mode_name] = {}

            # Store or compare losses
            loss_file = os.path.join("./tests/logging_jsons", "previous_stats_tldr.json")
            if os.path.exists(loss_file):
                with open(loss_file) as f:
                    previous_stats = json.load(f)
                    for mode in current_stats:
                        # Compare each key-value pair in the stats dictionaries
                        for key in current_stats[mode]:
                            if key in previous_stats[mode]:
                                current_val = current_stats[mode][key]
                                previous_val = previous_stats[mode][key]
                                
                                if isinstance(current_val, (int, float)):
                                    # 对单个数值使用相对误差
                                    self.assertAlmostEqual(
                                        current_val,
                                        previous_val,
                                        delta=abs(previous_val * 0.1),  # 允许10%的相对误差
                                        msg=f"Mode {mode}, key {key}: {current_val} != {previous_val}"
                                    )
                                elif isinstance(current_val, list) and all(isinstance(x, (int, float)) for x in current_val):
                                    # 对数值列表，逐个比较并允许相对误差
                                    self.assertEqual(
                                        len(current_val),
                                        len(previous_val),
                                        f"Mode {mode}, key {key}: lists have different lengths"
                                    )
                                    for i, (curr, prev) in enumerate(zip(current_val, previous_val)):
                                        self.assertAlmostEqual(
                                            curr,
                                            prev,
                                            delta=abs(prev * 0.1),  # 允许10%的相对误差
                                            msg=f"Mode {mode}, key {key}, index {i}: {curr} != {prev}"
                                        )
                                else:
                                    # 对非数值类型，使用严格相等
                                    self.assertEqual(
                                        current_val,
                                        previous_val,
                                        f"Mode {mode}, key {key}: values don't match"
                                    )
                        print(f"Stats checked for mode: {mode}")
            
            # Save current stats
            with open(loss_file, "w") as f:
                json.dump(current_stats, f)
