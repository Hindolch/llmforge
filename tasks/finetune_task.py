# tasks/finetune_task.py
from prefect import task
import subprocess

# tasks/finetune_task.py
# from prefect import task
# import subprocess

@task
def trigger_modal_finetune():
    # Step 1: Deploy the Modal finetune app
    subprocess.run(["modal", "deploy", "src/finetune.py"], check=True)

    # Step 2: Trigger finetune on entire HF dataset
    subprocess.run([
        "modal", "run", "src/finetune.py::run_finetune",
        "--model-name", "finetuned-tinyllama-lora"
    ], check=True)

    print("✅ Modal training job launched using HF dataset!")
