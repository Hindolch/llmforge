"""
Reddit Processing Task

This task cleans and filters the Reddit dataset using the `RedditProcessor`,
removes toxic/irrelevant content, and pushes the final dataset to Hugging Face Hub.
"""

from datetime import datetime
from src.data_processor import RedditProcessor
from prefect import task
import pandas as pd
import os

@task
def reddit_processor_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefect task to process Reddit data using Detoxify and text cleaning steps,
    then upload it to Hugging Face Hub.

    Args:
        df (pd.DataFrame): Raw Reddit data with 'text' and 'comments' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'prompt' and 'completion' fields.
    """
    processor = RedditProcessor(
        hf_repo_id="kenzi123/turboml_data",
        hf_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    df_cleaned = processor.process_and_save(df)
    return df_cleaned
