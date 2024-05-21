from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
from datetime import datetime

# Environment variables
os.environ["TOKEN"] = "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"
os.environ["HF_HOME"] = "/hf-cache"


# Formatting function for dataset
def formatting_func(example):
    text = f"### Vloga: Zdravstveni svetovalec. Vprašanje: {example['input']} Odgovor: {example['output']}"
    return text


# Model setup
def setup_model(checkpoint_path=None):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    download_directory = "/hf-cache"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            token=os.environ["TOKEN"],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=download_directory,
        )
    else:
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


# Load and format test data
def load_and_format_test_data(test_path="test.jsonl"):
    test_dataset = load_dataset("json", data_files=test_path, split="train")
    formatted_data = [formatting_func(example) for example in test_dataset]
    return formatted_data


# Generate predictions
def generate_predictions(model, tokenizer, formatted_data, max_length=1024):
    predictions = []
    model.eval()
    with torch.no_grad():
        for text in formatted_data:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(model.device)
            outputs = model.generate(**inputs)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
    return predictions


# Save predictions to a file
def save_predictions(predictions, output_path="predictions.jsonl"):
    with open(output_path, "w") as f:
        for pred in predictions:
            json.dump({"prediction": pred}, f)
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate predictions using a fine-tuned model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="test.jsonl",
        help="Path to the test data file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.jsonl",
        help="Path to save the predictions.",
    )

    args = parser.parse_args()

    model = setup_model(args.checkpoint_path)
    tokenizer = setup_tokenizer()
    formatted_data = load_and_format_test_data(args.test_path)
    predictions = generate_predictions(model, tokenizer, formatted_data)
    save_predictions(predictions, args.output_path)

    print(f"Predictions saved to {args.output_path}")
