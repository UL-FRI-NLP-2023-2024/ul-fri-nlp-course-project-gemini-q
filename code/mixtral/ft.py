from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import torch
import os
from qa_medical_dataloader import QADataloader, QADataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TOKEN"] = "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"
os.environ["HF_HOME"] = "/hf-cache"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def setup_model():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    download_directory = "/hf-cache"

    n4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=os.environ["TOKEN"],  # Updated to use 'token' instead of 'use_auth_token'
        quantization_config=n4_config,
        device_map="auto",
        cache_dir=download_directory,
    )

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return model


def setup_peft():
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        task_type="CAUSAL_LM",
    )

    return peft_config


def generate_response(prompt, model, tokenizer):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


def collate_fn(batch, tokenizer):
    print(f"Raw batch: {batch}")  # Print raw batch
    texts = [item["input_text"] for item in batch]
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print(f"Encoded batch: {encodings}")  # Print encoded batch
    return encodings


def train(model, peft_config):
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        print(torch.cuda.device_count())
        model.is_parallelizable = True
        model.model_parallel = True

    args = TrainingArguments(
        output_dir="/models/Mixtral_Med",  # Specify the output directory
        max_steps=1000,  # comment out this line if you want to train in epochs
        per_device_train_batch_size=32,
        warmup_steps=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=10,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2.5e-5,
        bf16=True,
    )

    max_seq_length = 1024

    # Load dataset
    dataset_path = "combine.json"
    qa_dataloader = QADataloader(dataset_path)
    train_dataloader, test_dataloader = qa_dataloader.get_dataloaders()

    tokenizer = qa_dataloader.get_tokenizer()

    # Check the first batch to ensure data is correct
    for batch in train_dataloader:
        print(f"First train batch: {batch}")
        break

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=test_dataloader.dataset if test_dataloader else None,
    )

    trainer.train()
    trainer.save_model(
        "/models/Mixtral_Med"
    )  # Save the model to the specified directory


if __name__ == "__main__":
    model = setup_model()
    peft_config = setup_peft()
    train(model, peft_config)
