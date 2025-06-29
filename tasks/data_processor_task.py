from datetime import datetime
from src.data_processor import RedditProcessor
from prefect import task
import pandas as pd
import subprocess
import os

@task
def reddit_processor_task(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"data/raw_reddit_{timestamp}.jsonl"
    processor = RedditProcessor(hf_repo_id="kenzi123/turboml_data", hf_token=os.getenv("HUGGINGFACE_TOKEN"))
    df_cleaned = processor.process_and_save(df)
    return df_cleaned


