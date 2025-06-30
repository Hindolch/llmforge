"""
Reddit Ingestion Module

This module defines an abstract base class and a concrete implementation for ingesting data
from Reddit using the PRAW API. It supports fetching post content and top comments from a given subreddit,
performing minimal cleaning and safe encoding, and returning the result as a pandas DataFrame.

"""

import os
import logging
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import praw
from dotenv import load_dotenv

from src.data_processor import remove_emojis

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding='utf-8')

# Load environment variables
load_dotenv()


class RedditIngestion(ABC):
    """
    Abstract base class defining the interface for any Reddit ingestion implementation.
    """
    @abstractmethod
    def fetch_posts(self, **kwargs):
        """
        Fetch posts from a data source.

        Returns:
            pd.DataFrame: DataFrame containing ingested data.
        """
        pass


@dataclass
class RedditIngestionConfig:
    """
    Configuration for Reddit ingestion process.
    """
    data_path: str = os.path.join("data", "raw_reddit.jsonl")


class RedditDataIngestion(RedditIngestion):
    """
    Concrete implementation of RedditIngestion using the PRAW API.

    Attributes:
        reddit (praw.Reddit): Reddit API client instance.
    """

    def __init__(self):
        """
        Initialize RedditDataIngestion with credentials from environment variables.
        """
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:llmforge-crawler:v1.0 (by u/Hindol007)"
        )

    def _safe_text_processing(self, text: str) -> str:
        """
        Safely process and clean input text to remove emojis and encoding issues.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned and safely encoded text.
        """
        if not text:
            return ""
        try:
            cleaned = remove_emojis(text)
            return cleaned.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            logging.warning(f"Encoding issue in text: {e}")
            return ''.join(c for c in text if ord(c) < 128)

    def fetch_posts(self, subreddit="learnmachinelearning", limit=120, max_comments=5) -> pd.DataFrame:
        """
        Fetch recent posts and top comments from a specified subreddit.

        Args:
            subreddit (str): Subreddit name to crawl.
            limit (int): Number of posts to fetch.
            max_comments (int): Number of top comments to include per post.

        Returns:
            pd.DataFrame: DataFrame containing post texts and their top comments.
        """
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
    """
    Factory class for instantiating ingestion strategies.
    """

    @staticmethod
    def get_ingestor(ingestor_type: str) -> RedditIngestion:
        """
        Retrieve an ingestion instance based on type.

        Args:
            ingestor_type (str): Type of ingestion ("reddit").

        Returns:
            RedditIngestion: Instance of ingestion strategy.

        Raises:
            ValueError: If the ingestor_type is unknown.
        """
        if ingestor_type == "reddit":
            return RedditDataIngestion()
        raise ValueError(f"Unknown ingestor type: {ingestor_type}")
