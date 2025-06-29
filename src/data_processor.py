#src/data_processor.py
import os, re, logging, subprocess
import pandas as pd
from detoxify import Detoxify
from datetime import datetime
from datasets import Dataset
from huggingface_hub import login as hf_login

from abc import ABC, abstractmethod

# -------------------- Utility Functions --------------------

def flatten_to_string(x):
    if isinstance(x, set):
        return "\n".join(sorted(str(i) for i in x))
    elif isinstance(x, dict):
        return "\n".join(f"{k}: {v}" for k, v in x.items())
    elif isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)

def strip_inline_json(text):
    return re.sub(r'\{.*?\}', '', text, flags=re.DOTALL).strip()

def clean_string_literals(text):
    return text.replace('\\"', '"').replace("\\n", "\n").strip()

def clean_text(text):
    # Replace bad formatting, trim, etc.
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

# -------------------- Processor Interface --------------------

class Processor(ABC):
    @abstractmethod
    def process_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------- Reddit Processor --------------------

from datasets import Dataset
from huggingface_hub import login as hf_login

class RedditProcessor(Processor):
    def __init__(self, hf_repo_id=None, hf_token=None):
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token

        if self.hf_token:
            hf_login(token=self.hf_token)

    def process_and_save(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df[df["text"].str.split().str.len() > 30]
        df["toxicity"] = Detoxify("original").predict(df["text"].tolist())["toxicity"]
        df = df[df["toxicity"] < 0.7]

        df["text"] = df["text"].apply(flatten_to_string)
        df["comments"] = df["comments"].apply(
            lambda x: [flatten_to_string(c) for c in x] if isinstance(x, list) else [flatten_to_string(x)]
        )

        df["prompt"] = df["text"].apply(lambda x:
        remove_emojis(clean_string_literals(strip_inline_json(clean_text(x))))
        )

        df["completion"] = df["comments"].apply(
            lambda x: remove_emojis(clean_string_literals(strip_inline_json(x[0])))
            if isinstance(x, list) and len(x) > 0 else ""
        )

        # Drop rows with empty completions
        df = df[df["completion"].str.strip().astype(bool)]

        df["prompt"] = df["prompt"].astype(str).apply(strip_inline_json).apply(clean_string_literals).apply(remove_emojis)
        df["completion"] = df["completion"].astype(str).apply(strip_inline_json).apply(clean_string_literals).apply(remove_emojis)


        # Push to Hugging Face if repo ID provided
        if self.hf_repo_id:
            hf_dataset = Dataset.from_pandas(df[["prompt", "completion"]])
            hf_dataset.push_to_hub(self.hf_repo_id)
            logging.info(f"📤 Uploaded to Hugging Face Hub: {self.hf_repo_id}")

        return df


# -------------------- Processor Factory --------------------
class ProcessorFactory:
    @staticmethod
    def get_processor(processor_type: str, out_path: str, hf_repo_id=None, hf_token=None) -> Processor:
        if processor_type == "reddit":
            return RedditProcessor(out_path, hf_repo_id=hf_repo_id, hf_token=hf_token)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
