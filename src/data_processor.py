import os
import logging
import pandas as pd
from detoxify import Detoxify
from src.text_cleaner import clean_text

class RedditProcessor:
    def __init__(self, out_path="data/raw_reddit.jsonl"):
        self.out_path = out_path

    def process_and_save(self, df):
        # ✅ Create parent dir if it doesn’t exist
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        df = df[df["text"].str.split().str.len() > 30]
        df["toxicity"] = Detoxify("original").predict(df["text"].tolist())["toxicity"]
        df = df[df["toxicity"] < 0.7]

        df["prompt"] = df["title"].apply(clean_text) + "\n\n" + df["text"].apply(clean_text)
        df["completion"] = df["comments"].apply(lambda x: "\n".join(x[:3]))

        df[["prompt", "completion"]].to_json(self.out_path, orient="records", lines=True)
        logging.info(f"✅ Saved cleaned Reddit data to {self.out_path}")
        return df

