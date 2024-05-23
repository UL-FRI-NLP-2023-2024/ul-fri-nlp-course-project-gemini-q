from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from datetime import datetime
import json

# Environment variables
os.environ["TOKEN"] = "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"
os.environ["HF_HOME"] = "/hf-cache"


# Dataset setup
def setup_datasets(train_path="train.jsonl", test_path="test.jsonl"):
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=test_path, split="train")
    return train_dataset, eval_dataset


# Formatting function for dataset
def formatting_func(example):
    text = f"### Vloga: Zdravstveni svetovalec. Vprašanje: {example['input']} Odgovor: {example['output']}"
    return text


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


# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Setup LoRA model
def get_lora_model(model):
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "w1",
            "w2",
            "w3",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model


# Training function
def train():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    model = setup_model()
    tokenizer = setup_tokenizer()

    def generate_and_tokenize_prompt(prompt):
        max_length = 1024
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset, eval_dataset = setup_datasets()
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_lora_model(model)

    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        print(f"Using {torch.cuda.device_count()} GPUs")
        model.is_parallelizable = True
        model.model_parallel = True

    project = "medical-finetune"
    base_model_name = "mixtral8x7b"
    run_name = base_model_name + "-" + project
    output_dir = "/models/" + run_name
    os.makedirs(output_dir, exist_ok=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=50,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            max_steps=1000,
            learning_rate=2e-5,  # Adjusted learning rate for fine-tuning
            fp16=True,
            optim="adamw_torch",
            logging_steps=50,  # When to start reporting loss
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=200,  # Save checkpoints every 200 steps
            eval_strategy="steps",  # Evaluate the model every logging step
            eval_steps=200,  # Evaluate and save checkpoints every 200 steps
            do_eval=True,  # Perform evaluation at the end of training
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    return model, tokenizer


# Prediction function
def predict(model, tokenizer, test_dataset_path="test.jsonl", output_dir="predictions"):
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
    trained_model, trained_tokenizer = train()
    predict(trained_model, trained_tokenizer)
