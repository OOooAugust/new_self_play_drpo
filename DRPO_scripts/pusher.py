# push local model to huggingface
# from huggingface_hub import push_to_hub
from transformers import AutoModelForCausalLM, AutoTokenizer

output_dir = "D:/Downloads/results/tldr/gpm-2dim-088004/"
model_id = "Eehan/pythia-1b-drpo-1e-gpm-temp-0.88-beta-0.04"

model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

