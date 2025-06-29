# src/data_ingestion.py
import os
from dataclasses import dataclass
import praw
import pandas as pd
import logging
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RedditIngestion(ABC):
    @abstractmethod
    def fetch_posts(self, **kwargs):
        pass


@dataclass
class RedditIngestionConfig:
    data_path: str = os.path.join("data", "raw_reddit.jsonl")


class RedditDataIngestion(RedditIngestion):
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="llmforge-crawler"
        )

    def fetch_posts(self, subreddit="learnmachinelearning", limit=120, max_comments=5):
        posts = []
        for post in self.reddit.subreddit(subreddit).new(limit=limit):
            if post.selftext:
                post.comments.replace_more(limit=0)
                title = post.title.encode("utf-8", errors="ignore").decode("utf-8")
                text = post.selftext.encode("utf-8", errors="ignore").decode("utf-8")
                comments = [
                    c.body.strip().encode("utf-8", errors="ignore").decode("utf-8")
                    for c in post.comments[:max_comments] if len(c.body.strip()) > 10
                ]

                posts.append({
                    "text": text,
                    "comments": comments
                })

        return pd.DataFrame(posts)

class IngestorFactory:
    @staticmethod
    def get_ingestor(ingestor_type: str) -> RedditIngestion:
        if ingestor_type == "reddit":
            return RedditDataIngestion()
        else:
            raise ValueError(f"Unknown ingestor type: {ingestor_type}")