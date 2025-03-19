from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class DRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`OnlineDRPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse]

    Parameters:
        learning_rate (`float`, *optional*, defaults to `5e-7`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        preference_model_path (`str` or `None`, *optional*, defaults to `None`):
            Path to the preference model. Either `judge` or `reward_model_path` must be set, but not both.
        max_new_tokens `int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        max_length (`int`, *optional*, defaults to `512`):
            Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If
            the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the
            completion as possible.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`float`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to
            encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be
            a positive value.
        beta(`float` or `list[float]` *optional*, defaults to `0.1`)
            Parameter controlling the deviation from the reference model. Higher β means less deviation from
            the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter of KL divergence.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout. #FIXME: This is not yet implemented.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `False`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
    """

    learning_rate: float = field(
        default=5e-7,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of"
            "transformers.TrainingArguments."
        }
    )

    preference_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the preference model."
        }
    )

    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If "
            "the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the "
            "completion as possible."
        },
    )

    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    
    missing_eos_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    
    beta: list[float] = field(
        default_factory=lambda: [0.1],
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter denoted by "
            "τ in the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β "
            "is selected for each new epoch and the last β is used for the rest of the epochs."
        },
    )

    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )

    ds_gather_for_generation: bool = field(
        default=True,
        meatadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,"
            "improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU,"
            "albeit at the cost of slower generation."
        },
    )

    num_star: int = field(
        defaul=1,
        metadata={
            "help": "Number of newly generated completions to compare with the reference model."
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.beta, "__len__") and len(self.beta) == 1:
            self.beta = self.beta[0]
    


