"""
Data Processing Module

This module provides utilities to process raw Reddit data, clean text,
evaluate toxicity using Detoxify, and upload processed datasets to the Hugging Face Hub.

"""

import os
import re
import logging
import pandas as pd
from abc import ABC, abstractmethod
from detoxify import Detoxify
from datasets import Dataset
from huggingface_hub import login as hf_login
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")


def flatten_to_string(x):
    """
    Flattens common nested types (set, dict, list) into a single string.
    """
    if isinstance(x, set):
        return "\n".join(sorted(str(i) for i in x))
    elif isinstance(x, dict):
        return "\n".join(f"{k}: {v}" for k, v in x.items())
    elif isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)


def strip_inline_json(text):
    """
    Removes inline JSON-like content from text.
    """
    return re.sub(r'\{.*?\}', '', text, flags=re.DOTALL).strip()


def clean_string_literals(text):
    """
    Cleans escaped quotes and newline characters from text.
    """
    return text.replace('\\"', '"').replace("\\n", "\n").strip()


def clean_text(text):
    """
    Removes URLs and excess whitespace from text.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_emojis(text):
    """
    Removes emoji and special unicode characters from text.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


class Processor(ABC):
    """
    Abstract base class for processing raw data into a usable format.
    """
    @abstractmethod
    def process_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame and optionally saves/uploads the result.

        Args:
            df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        pass



class RedditProcessor(Processor):
    """
    Reddit-specific implementation of Processor.

    - Filters out short posts
    - Removes toxic content using Detoxify
    - Cleans prompt/comment text
    - Optionally pushes to Hugging Face Hub
    """

    def __init__(self, hf_repo_id=None, hf_token=None):
        """
        Initialize RedditProcessor.

        Args:
            hf_repo_id (str): Hugging Face repo ID to push processed data to.
            hf_token (str): Hugging Face token for authentication.
        """
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token

        if self.hf_token:
            hf_login(token=self.hf_token)

    def process_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function:
        - Filters by length and toxicity
        - Extracts and cleans prompt and completion fields
        - Uploads processed data to Hugging Face if configured

        Args:
            df (pd.DataFrame): Raw Reddit data with 'text' and 'comments' columns.

        Returns:
            pd.DataFrame: Processed dataset with 'prompt' and 'completion'.
        """
        # Basic filters
        df = df[df["text"].str.split().str.len() > 30]
        df["toxicity"] = Detoxify("original").predict(df["text"].tolist())["toxicity"]
        df = df[df["toxicity"] < 0.7]

        # String flattening for raw text
        df["text"] = df["text"].apply(flatten_to_string)
        df["comments"] = df["comments"].apply(
            lambda x: [flatten_to_string(c) for c in x] if isinstance(x, list) else [flatten_to_string(x)]
        )

        # Generate prompt and completion
        df["prompt"] = df["text"].apply(lambda x:
            remove_emojis(clean_string_literals(strip_inline_json(clean_text(x))))
        )

        df["completion"] = df["comments"].apply(
            lambda x: remove_emojis(clean_string_literals(strip_inline_json(x[0])))
            if isinstance(x, list) and len(x) > 0 else ""
        )

        # Drop rows with empty completions
        df = df[df["completion"].str.strip().astype(bool)]

        # Final clean
        df["prompt"] = df["prompt"].astype(str).apply(strip_inline_json).apply(clean_string_literals).apply(remove_emojis)
        df["completion"] = df["completion"].astype(str).apply(strip_inline_json).apply(clean_string_literals).apply(remove_emojis)

        # Upload to HF Hub
        if self.hf_repo_id:
            hf_dataset = Dataset.from_pandas(df[["prompt", "completion"]])
            hf_dataset.push_to_hub(self.hf_repo_id)
            logging.info(f"📤 Uploaded to Hugging Face Hub: {self.hf_repo_id}")

        return df



class ProcessorFactory:
    """
    Factory class to instantiate data processors by type.
    """

    @staticmethod
    def get_processor(processor_type: str, out_path: str, hf_repo_id=None, hf_token=None) -> Processor:
        """
        Factory method to retrieve appropriate processor instance.

        Args:
            processor_type (str): Type of processor ('reddit').
            out_path (str): Output path (currently unused, placeholder for interface consistency).
            hf_repo_id (str): Optional Hugging Face repo ID.
            hf_token (str): Optional Hugging Face token.

        Returns:
            Processor: Instance of a Processor subclass.
        """
        if processor_type == "reddit":
            return RedditProcessor(hf_repo_id=hf_repo_id, hf_token=hf_token)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
