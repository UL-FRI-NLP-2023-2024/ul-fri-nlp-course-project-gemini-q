from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json

# Environment variables
os.environ["TOKEN"] = "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"
os.environ["HF_HOME"] = "/hf-cache"


# Model setup
def setup_model():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    download_directory = "/hf-cache"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=os.environ["TOKEN"],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=download_directory,
    )

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return model


# Tokenizer setup
def setup_tokenizer():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    download_directory = "/hf-cache"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["TOKEN"],
        device_map="auto",
        cache_dir=download_directory,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )

    tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to eos_token

    return tokenizer


# Prediction function
def predict(
    model, tokenizer, test_dataset_path="test.jsonl", output_dir="predictions_vanilla"
):
    test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")

    def generate_prompt(example):
        return f"### Vloga: Zdravstveni svetovalec. Vprašanje: {example['input']}"

    os.makedirs(output_dir, exist_ok=True)

    for idx, example in enumerate(test_dataset):
        prompt = generate_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=1024, num_beams=5, early_stopping=True
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_dict = {"input": example["input"], "prediction": prediction}

        with open(os.path.join(output_dir, f"prediction_{idx}.json"), "w") as f:
            json.dump(prediction_dict, f)


if __name__ == "__main__":
    model = setup_model()
    tokenizer = setup_tokenizer()
    predict(model, tokenizer)
