from datetime import datetime
from src.data_processor import RedditProcessor
from prefect import task
import pandas as pd

@task
def reddit_processor_task(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    processor = RedditProcessor(out_path=f"data/raw_reddit_{timestamp}.jsonl")
    df_cleaned = processor.process_and_save(df)
    return df_cleaned
