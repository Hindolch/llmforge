import modal
from modal import App, Image, Secret
import os

app = App("finetune-tinyllama-lora")

image = Image.debian_slim().pip_install(
    "transformers", "datasets", "torch", "accelerate", "huggingface_hub", 
    "peft", "bitsandbytes"
)
@app.function(
    gpu="T4",
    timeout=60 * 60,
    image=image,
    secrets=[Secret.from_name("huggingface-token")]
)
def run_finetune(model_name: str = "finetuned-tinyllama-lora"):
    import os
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

    # Login
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    login(token=hf_token)

    # Load dataset from HF
    print("📦 Loading dataset from Hugging Face Hub...")
    dataset = load_dataset("kenzi123/turboml_data", split="train")

    # Filter to ensure validity
    dataset = dataset.filter(
        lambda x: isinstance(x["prompt"], str) and isinstance(x["completion"], str)
    )

    # Load model & tokenizer
    print("📦 Loading TinyLlama model...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Ensure special tokens
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    print("⚙️ Setting up LoRA configuration...")
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

    # Apply LoRA
    print("🔧 Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenization
    def tokenize_example(example):
        prompt = example["prompt"].strip()
        completion = example["completion"].strip()
        full_text = prompt + tokenizer.eos_token + completion + tokenizer.eos_token
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing data"
    )

    # Training args
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print("🚀 Starting LoRA training...")
    trainer.train()

    print("💾 Saving LoRA adapters...")
    model.save_pretrained("/tmp/output")
    tokenizer.save_pretrained("/tmp/output")

    print(f"📤 Pushing LoRA model to Hugging Face Hub: {model_name}")
    trainer.push_to_hub(commit_message="Fine-tuned TinyLlama with LoRA")

    print(f"✅ LoRA model fine-tuned and pushed to: https://huggingface.co/{model_name}")
    return f"https://huggingface.co/{model_name}"
