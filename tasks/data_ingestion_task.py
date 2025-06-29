from dotenv import load_dotenv
import os
import logging
from src.data_ingestion import RedditDataIngestion

from prefect import task

@task
def reddit_ingestion_task():
    # Force dotenv load
    dotenv_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Log for debug
    client_id = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_SECRET")
    if not client_id or not secret:
        raise ValueError("❌ Reddit credentials not found — ensure .env is loaded correctly.")

    print("✅ Reddit creds loaded inside task.")
    
    ingestor = RedditDataIngestion()  # will use env vars now
    df = ingestor.fetch_posts()
    
    if df.empty:
        logging.warning("⚠️ No posts were ingested — likely due to encoding errors or filters.")
    return df
