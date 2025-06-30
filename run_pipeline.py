"""
Main Prefect Pipeline for Reddit Data Ingestion and Fine-tuning

This script defines the full Prefect pipeline to:
1. Ingest posts + comments from Reddit
2. Clean + filter data for LLM fine-tuning
3. Trigger Modal fine-tuning on HF dataset

Upon start, a notification email is sent using the email alert module.
"""

from prefect import flow
from tasks.data_ingestion_task import reddit_ingestion_task
from tasks.data_processor_task import reddit_processor_task
from tasks.finetune_task import trigger_modal_finetune
from src.email import send_email_alert

import os
from dotenv import load_dotenv

@flow
def reddit_ingestion_flow():
    """Main orchestrated flow to collect, process, and fine-tune Reddit data using Modal."""
    load_dotenv()

    send_email_alert("🚀 LLMForge Pipeline Triggered")

    # Step 1: Ingest Reddit Posts
    df = reddit_ingestion_task()
    print("✅ Reddit Ingestion Complete:\n", df.head())

    # Step 2: Process + Clean Posts
    df_cleaned = reddit_processor_task(df)
    print("✅ Reddit Processing Complete:\n", df_cleaned.head())

    # Step 3: Trigger Modal Finetuning
    trigger_modal_finetune.submit()
    print("✅ Modal Finetune Triggered ✅")

if __name__ == "__main__":
    reddit_ingestion_flow()
