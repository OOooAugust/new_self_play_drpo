from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
from pathlib import Path

model_dir = "./output/3"
model_name = "Eehan/Qwen2-0.5B-drpo-imdb-default-3"


model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)