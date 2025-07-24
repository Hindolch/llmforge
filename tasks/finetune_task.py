#tasks/finetune_task.py
"""
Finetune Task - Launches TinyLlama LoRA training on Modal

This Prefect task deploys the Modal app and triggers the finetuning function
that trains a TinyLlama model using the HF dataset and uploads the LoRA weights.
"""

from prefect import task
import subprocess

@task
def trigger_modal_finetune():
    """
    Deploys the Modal LoRA finetuning app and runs the training function remotely.

    Raises:
        subprocess.CalledProcessError: If the deployment or run command fails.
    """
    try:
        # 🔧 Step 1: Deploy the Modal app
        subprocess.run(["modal", "deploy", "src/finetune.py"], check=True)

        # 🚀 Step 2: Run the remote finetune function
        subprocess.run([
            "modal", "run", "src/finetune.py::run_finetune",
            "--model-name", "finetuned-tinyllama-lora"
        ], check=True)

        print("✅ Modal training job launched using HF dataset!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Modal job failed: {e}")
        raise
