from prefect import flow
from tasks.data_ingestion_task import reddit_ingestion_task
from tasks.data_processor_task import reddit_processor_task
from tasks.finetune_task import trigger_modal_finetune
from src.email import send_email_alert
import os
import sys
import io
from dotenv import load_dotenv

@flow
def reddit_ingestion_flow():
    load_dotenv()

    send_email_alert("Pipeline Triggered")

    df = reddit_ingestion_task()
    print("✅ Reddit Ingestion Flow complete:\n", df.head())

    df_cleaned = reddit_processor_task(df)
    print("✅ Reddit Processing Flow complete:\n", df_cleaned.head())

    # ✅ Call the Modal finetune task
    trigger_modal_finetune.submit()
    print("✅ Modal finetune task triggered")



if __name__ == "__main__":
    reddit_ingestion_flow()
