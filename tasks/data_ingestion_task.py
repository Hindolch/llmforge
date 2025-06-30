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
from src.data_ingestion import RedditDataIngestion

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
