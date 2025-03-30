from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, pipeline

from transformers import pipeline
tokenizer_3 = AutoTokenizer.from_pretrained("Eehan/Qwen2-0.5B-drpo-imdb-default-3")
tokenizer_3.padding_side = "left"
tokenizer_3.add_special_tokens({"pad_token": "[PAD]"})

tokenizer_4 = AutoTokenizer.from_pretrained("Eehan/Qwen2-0.5B-drpo-imdb-indifferent-4")
tokenizer_4.padding_side = "left"
tokenizer_4.add_special_tokens({"pad_token": "[PAD]"})


pipe_3 = pipeline(
    "text-generation",
    model="Eehan/Qwen2-0.5B-drpo-imdb-default-3",
    tokenizer=tokenizer_3,
    batch_size=128,
    eos_token_id=tokenizer_3.eos_token_id,
)
pipe_4 = pipeline(
    "text-generation",
    model="Eehan/Qwen2-0.5B-drpo-imdb-indifferent-4",
    tokenizer=tokenizer_4,
    batch_size=128,
     eos_token_id=tokenizer_4.eos_token_id,
)

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

def extract_dialogue(examples: dict) -> dict:
    prompts = examples["prompt"]
    
    default_3_list = []
    indifferent_4_list = []
    score_default_3_list = []
    score_indifferent_4_list = []
    generated_3 = pipe_3(
        prompts,
        max_new_tokens=256,
        eos_token_id=tokenizer_3.eos_token_id,
    )
    generated_4 = pipe_4(
        prompts,
        max_new_tokens=256,
        eos_token_id=tokenizer_4.eos_token_id,
    )
    flat_generated = []

    flat_generated =[batch[0]["generated_text"] for batch in generated_3]
    flat_generated.extend([batch[0]["generated_text"] for batch in generated_4])
    sentiment_analysis_result = sentiment_analysis(flat_generated, batch_size=256, truncation=True, padding=True)
    for i in range(len(sentiment_analysis_result)//2):
        res1 = sentiment_analysis_result[i]
        res2 = sentiment_analysis_result[len(sentiment_analysis_result)//2+i]
        text1 = flat_generated[i]
        text2 = flat_generated[len(sentiment_analysis_result)//2+i]

        if res1["label"] == "NEGATIVE":
            score_3 = 1 - res1["score"]
        else:
            score_3 = res1["score"]
        if res2["label"] == "NEGATIVE":
            score_4 = 1 - res2["score"]
        else:
            score_4 = res2["score"]
        default_3_list.append(text1)
        indifferent_4_list.append(text2)
        score_default_3_list.append(score_3)
        score_indifferent_4_list.append(score_4)
        
    return {
        "default_3_completion": default_3_list,
        "indifferent_4_completion": indifferent_4_list,
        "score_default_3": score_default_3_list,
        "score_indifferent_4": score_indifferent_4_list,
    }   




if __name__ == "__main__":


    dataset = load_dataset("Kyleyee/eval_data_imdb_with_subsft")["temperature_0"]
    dataset = dataset.remove_columns(["dpo","drdpo",'dpo_scores','drdpo_scores',
                                      'drdpo_with_estimate_preference', 'drdpo_with_estimate_preference_scores',
                                      'drdpo_with_wrong_preference', 'drdpo_with_wrong_preference_scores',
                                      'drdpo_with_subsft', 'drdpo_with_subsft_scores',])
    # dataset = dataset.select(range(2500))
    print(f"✅ Applied chat template: {dataset[0]}")

    dataset = dataset.map(extract_dialogue, batched=True, batch_size=128)

    

   
    dataset.push_to_hub("Eehan/eval-imdb-drpo-3-drpo-4-1000")
    