"""
Fine-tuning TinyLlama with LoRA using Modal

This script runs inside a Modal container with GPU access to fine-tune the
'TinyLlama/TinyLlama-1.1B-Chat-v1.0' model using LoRA adapters and a
custom Hugging Face dataset.

"""

import os
import modal
from modal import App, Image, Secret


app = App("finetune-tinyllama-lora")

image = Image.debian_slim().pip_install(
    "transformers",
    "datasets",
    "torch",
    "accelerate",
    "huggingface_hub",
    "peft",
    "bitsandbytes"
)


@app.function(
    gpu="T4",
    timeout=60 * 60,
    image=image,
    secrets=[Secret.from_name("huggingface-token")]
)
def run_finetune(model_name: str = "finetuned-tinyllama-lora"):
    """
    Run LoRA fine-tuning on TinyLlama and push model to Hugging Face Hub.

    Args:
        model_name (str): Target model name to be pushed on Hugging Face.

    Returns:
        str: Hugging Face model URL
    """
    import torch
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoTokenizer,
        AutoModelForCausalLM,
        default_data_collator
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    from huggingface_hub import login

    # 🔐 Authenticate with Hugging Face
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    login(token=hf_token)

    # 📦 Load dataset from Hugging Face Hub
    print("📦 Loading dataset from Hugging Face Hub...")
    dataset = load_dataset("kenzi123/turboml_data", split="train")

    # Filter out invalid rows
    dataset = dataset.filter(
        lambda x: isinstance(x["prompt"], str) and isinstance(x["completion"], str)
    )

    # 📥 Load base model and tokenizer
    print("📥 Loading TinyLlama base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Add EOS and PAD token if missing
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🔧 Setup LoRA adapter configuration
    print("🔧 Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 🧪 Tokenize prompt + completion pairs
    def tokenize_example(example):
        prompt = example["prompt"].strip()
        completion = example["completion"].strip()
        full_text = prompt + tokenizer.eos_token + completion + tokenizer.eos_token

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing data"
    )

    # 🧠 Define training arguments
    args = TrainingArguments(
        output_dir="/tmp/output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="/tmp/logs",
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        push_to_hub=True,
        hub_model_id=model_name,
        hub_token=hf_token,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # 🏋️ Run training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()

    # 💾 Save artifacts
    print("💾 Saving model & tokenizer...")
    model.save_pretrained("/tmp/output")
    tokenizer.save_pretrained("/tmp/output")

    # ⬆️ Push model to HF Hub
    print(f"📤 Pushing model to Hugging Face Hub: {model_name}")
    trainer.push_to_hub(commit_message="Fine-tuned TinyLlama with LoRA")

    print(f"✅ Successfully pushed to: https://huggingface.co/{model_name}")
    return f"https://huggingface.co/{model_name}"
