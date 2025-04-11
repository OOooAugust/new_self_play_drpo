# push local model to huggingface
# from huggingface_hub import push_to_hub
from transformers import AutoModelForCausalLM, AutoTokenizer

output_dir = "./output/tldr/0/"
model_id = "Eehan/pythia-1b-deduped-tldr-ppo-0.9tmp"

model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

