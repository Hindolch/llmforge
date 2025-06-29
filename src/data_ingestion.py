import os
import logging
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import praw
from dotenv import load_dotenv

from src.data_processor import remove_emojis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding='utf-8')


load_dotenv()  # ✅ Load env vars early


class RedditIngestion(ABC):
    @abstractmethod
    def fetch_posts(self, **kwargs):
        pass

@dataclass
class RedditIngestionConfig:
    data_path: str = os.path.join("data", "raw_reddit.jsonl")

class RedditDataIngestion(RedditIngestion):
    def __init__(self):
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        self.reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent="script:llmforge-crawler:v1.0 (by u/Hindol007)"
    )

    def _safe_text_processing(self, text):
        if not text:
            return ""
        try:
            cleaned = remove_emojis(text)
            return cleaned.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            logging.warning(f"Encoding issue in text: {e}")
            return ''.join(c for c in text if ord(c) < 128)


    def fetch_posts(self, subreddit="learnmachinelearning", limit=10, max_comments=5):
        posts = []

        try:
            for post in self.reddit.subreddit(subreddit).new(limit=limit):
                if post.selftext and isinstance(post.selftext, str):
                    post.comments.replace_more(limit=0)

                    text = self._safe_text_processing(post.selftext)
                    comments = [
                        self._safe_text_processing(c.body.strip())
                        for c in post.comments[:max_comments]
                        if len(self._safe_text_processing(c.body.strip())) > 10
                    ]

                    posts.append({"text": text, "comments": comments})

        except Exception as e:
            logging.error(f"Error during Reddit data ingestion: {e}")

        if not posts:
            logging.warning("⚠️ No posts were ingested — likely due to encoding errors or filters.")

        return pd.DataFrame(posts)

class IngestorFactory:
    @staticmethod
    def get_ingestor(ingestor_type: str) -> RedditIngestion:
        if ingestor_type == "reddit":
            return RedditDataIngestion()
        raise ValueError(f"Unknown ingestor type: {ingestor_type}")
