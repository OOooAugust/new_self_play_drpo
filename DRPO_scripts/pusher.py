# push local model to huggingface
# from huggingface_hub import push_to_hub
from transformers import AutoModelForCausalLM, AutoTokenizer

output_dir = "../results/hh/066004/3000"
model_id = "Eehan/Qwen2.5-1.5B-drpo-hh-0.82e-temp-0.66-beta-0.04"

model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

