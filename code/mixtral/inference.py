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
from peft import PeftModel, PeftConfig

# Environment variables
os.environ["TOKEN"] = "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"
os.environ["HF_HOME"] = "/hf-cache"


# Formatting function for dataset (inference)
def formatting_func_inference(example):
    text = f"### Vloga: Zdravstveni svetovalec. Vprašanje: {example['input']} Odgovor:"
    return text


# Model setup
def setup_model(checkpoint_path=None, adapter_path=None):
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

    if adapter_path:
        config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, config=config)

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
        padding_side="left",  # Ensure left padding for correct generation
        add_eos_token=True,
        add_bos_token=True,
    )

    tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to eos_token

    return tokenizer


# Load and format test data
def load_and_format_test_data(test_path="test.jsonl"):
    test_dataset = load_dataset("json", data_files=test_path, split="train")
    formatted_data = [formatting_func_inference(example) for example in test_dataset]
    return formatted_data


# Generate predictions with batching and immediate write to file
def generate_predictions(
    model,
    tokenizer,
    formatted_data,
    output_path="predictions.jsonl",
    max_length=1024,
    batch_size=8,
):
    model.eval()
    with torch.no_grad():
        with open(output_path, "w") as f:
            for i in range(0, len(formatted_data), batch_size):
                batch_texts = formatted_data[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                ).to(model.device)
                outputs = model.generate(**inputs, max_length=max_length)
                batch_predictions = [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                for prediction in batch_predictions:
                    json.dump({"prediction": prediction}, f)
                    f.write("\n")
    print(f"Predictions saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate predictions using a fine-tuned model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the adapter model weights.",
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

    model = setup_model(
        checkpoint_path=args.checkpoint_path, adapter_path=args.adapter_path
    )
    tokenizer = setup_tokenizer()
    formatted_data = load_and_format_test_data(args.test_path)
    generate_predictions(model, tokenizer, formatted_data, output_path=args.output_path)

    print(f"Predictions saved to {args.output_path}")
