from prefect import task
from src.data_ingestion import RedditDataIngestion
import pandas as pd
import logging

@task
def reddit_ingestion_task() -> pd.DataFrame:
    ingestor = RedditDataIngestion()
    df_raw = ingestor.fetch_posts()
    return df_raw