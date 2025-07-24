"""
Universal Ingestion Module

This module defines abstract base classes and concrete implementations for ingesting data
from Reddit or user-provided `.jsonl` files. Each ingestor returns cleaned, structured
data as a pandas DataFrame for use in fine-tuning or training pipelines.
"""

import os
import json
import logging
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import praw
from dotenv import load_dotenv

from .data_processor import remove_emojis


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding='utf-8')
load_dotenv()

### --------------------------- Reddit Ingestion ---------------------------

class RedditIngestion(ABC):
    """
    Abstract base class defining the interface for Reddit ingestion.
    """
    @abstractmethod
    def fetch_posts(self, **kwargs):
        pass


@dataclass
class RedditIngestionConfig:
    data_path: str = os.path.join("data", "raw_reddit.jsonl")


class RedditDataIngestion(RedditIngestion):
    """
    Ingests posts + top comments from a subreddit using the PRAW API.
    """
    def __init__(self):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:llmforge-crawler:v1.0 (by u/Hindol007)"
        )

    def _safe_text_processing(self, text: str) -> str:
        if not text:
            return ""
        try:
            cleaned = remove_emojis(text)
            return cleaned.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            logging.warning(f"Encoding issue in text: {e}")
            return ''.join(c for c in text if ord(c) < 128)

    def fetch_posts(self, subreddit="learnmachinelearning", limit=120, max_comments=5) -> pd.DataFrame:
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
            logging.error(f"Error during Reddit ingestion: {e}")

        if not posts:
            logging.warning("⚠️ No posts were ingested — possibly empty subreddit or errors.")
        return pd.DataFrame(posts)


### --------------------------- JSONL Ingestion ---------------------------

class BaseIngestion(ABC):
    """
    Abstract base class for ingestion from user-uploaded JSONL datasets.
    """
    @abstractmethod
    def fetch_data(self, **kwargs):
        pass


@dataclass
class JSONLIngestionConfig:
    file_path: str


class UserJSONLIngestion(BaseIngestion):
    def __init__(self, config: JSONLIngestionConfig):
        self.file_path = config.file_path

    def _convert_json_to_records(self) -> list:
        """
        Converts a .json file to a list of records (dicts), suitable for DataFrame loading.
        Supports both `.json` and `.jsonl`.
        """
        if self.file_path.endswith(".jsonl"):
            with open(self.file_path, "r") as f:
                return [json.loads(line) for line in f if line.strip()]

        elif self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):  # Some JSONs are just a dict
                    data = [data]
                return data

        raise ValueError("❌ Unsupported file format. Please provide `.json` or `.jsonl`.")

    def fetch_data(self) -> pd.DataFrame:
        """
        Loads the JSON/JSONL file into a pandas DataFrame.
        """
        records = self._convert_json_to_records()
        df = pd.DataFrame(records)
        print(f"📥 Loaded {len(df)} records from {self.file_path}")
        return df


### --------------------------- Ingestor Factory ---------------------------

class IngestorFactory:
    """
    Factory for dynamic ingestion strategy selection.
    """

    @staticmethod
    def get_ingestor(ingestor_type: str, **kwargs):
        """
        Args:
            ingestor_type (str): "reddit" or "jsonl"
            kwargs: Parameters like file_path, config, etc.
        """
        if ingestor_type == "reddit":
            return RedditDataIngestion()
        elif ingestor_type == "json":
            if "file_path" not in kwargs:
                raise ValueError("file_path is required for JSONL ingestion.")
            config = JSONLIngestionConfig(file_path=kwargs["file_path"])
            return UserJSONLIngestion(config)
        else:
            raise ValueError(f"Unknown ingestor type: {ingestor_type}")

# # Reddit usage
# reddit_ingestor = IngestorFactory.get_ingestor("reddit")
# df_reddit = reddit_ingestor.fetch_posts()

# # JSONL usage
# jsonl_ingestor = IngestorFactory.get_ingestor("jsonl", file_path="data/my_dataset.jsonl")
# df_jsonl = jsonl_ingestor.fetch_data()
