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
        use_auth_token=os.environ["TOKEN"],  # Add authentication token here
        quantization_config=n4_config,
        device_map="auto",
        cache_dir=download_directory,
        attn_implementation="flash_attention_2",
    )

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return model


def setup_tokenizer():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_auth_token=os.environ["TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


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


def generate_response(prompt, model):
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


def train(model, peft_config, tokenizer):
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        print(torch.cuda.device_count())
        model.is_parallelizable = True
        model.model_parallel = True

    args = TrainingArguments(
        output_dir="Mixtral_Med",
        # num_train_epochs=5,
        max_steps=1000,  # comment out this line if you want to train in epochs
        per_device_train_batch_size=32,
        warmup_steps=0.03,
        logging_steps=10,
        save_strategy="epoch",
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=10,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2.5e-5,
        bf16=True,
        # lr_scheduler_type='constant',
    )

    max_seq_length = 1024

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=args,
        # train_dataset=instruct_tune_dataset["train"],
        # eval_dataset=instruct_tune_dataset["test"]
    )

    trainer.train()
    trainer.save_model("Mixtral_Med")


if __name__ == "__main__":
    pass
