"""
Reddit Data Ingestion Task 

This task loads environment variables securely, authenticates with Reddit via `praw`,
and fetches a filtered set of subreddit posts using `RedditDataIngestion`.

Returns a DataFrame containing post texts and comments.

"""

import os
import logging
from dotenv import load_dotenv
from prefect import task
from src.data_ingestion import RedditDataIngestion, UserJSONLIngestion, JSONLIngestionConfig
import os
import json
import pandas as pd
from datasets import Dataset

@task
def reddit_ingestion_task():
    """
    Prefect task to ingest Reddit data via the RedditDataIngestion class.

    This function:
    - Loads environment variables from a `.env` file
    - Initializes a Reddit ingestor via PRAW
    - Fetches filtered Reddit posts
    - Returns the dataset as a pandas DataFrame

    Returns:
        pd.DataFrame: Ingested and filtered Reddit posts
    """
    # 🔐 Load environment variables
    dotenv_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)

    client_id = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not secret:
        raise ValueError("❌ Reddit credentials not found — ensure `.env` is loaded correctly.")

    print("✅ Reddit credentials loaded successfully inside Prefect task.")

    # 🚀 Fetch posts via RedditDataIngestion
    ingestor = RedditDataIngestion()
    df = ingestor.fetch_posts()

    if df.empty:
        logging.warning("⚠️ No posts were ingested — likely due to encoding issues or no valid data.")

    return df


@task
def user_jsonl_ingestion_task(file_path: str) -> pd.DataFrame:
    """
    Ingest user-provided JSON or JSONL dataset into a DataFrame, handling conversion in-memory.

    Args:
        file_path (str): Path to the dataset file (.json or .jsonl)

    Returns:
        pd.DataFrame: Parsed data ready for downstream use
    """
    ext = os.path.splitext(file_path)[1]

    if ext == ".jsonl":
        with open(file_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        print(f"✅ Loaded {len(lines)} records from JSONL")
        return pd.DataFrame(lines)

    elif ext == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)

        # If it's a dict, wrap into list
        if isinstance(data, dict):
            data = [data]

        print(f"✅ Loaded {len(data)} records from JSON")
        return pd.DataFrame(data)

    else:
        raise ValueError("❌ Unsupported file format. Please provide a .json or .jsonl file.")
