# src/data_ingestion.py
from dataclasses import dataclass
import praw, re, os
import pandas as pd
import logging
import os

@dataclass
class RedditIngestionConfig:
    data_path: str = os.path.join("data", "raw_reddit.jsonl")

class RedditDataIngestion:
    def __init__(self):
        self.config = RedditIngestionConfig()
        self.reddit = praw.Reddit(
            # client_id=os.getenv("REDDIT_CLIENT_ID"),
            # client_secret=os.getenv("REDDIT_SECRET"),
            # user_agent="llmforge-crawler"
            client_id="o6Z84Z0rxget-8ybCfVSnQ",             # 👈 this is your "client_id"
            client_secret="vkGVNxboxgpuQC2QKqJ--9nRB6mNYw",  # 👈 this is your "client_secret"
            user_agent="llmforge-crawler"
        )

    def fetch_posts(self, subreddit="learnmachinelearning", limit=50, max_comments=5):
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

                posts.append({   # 🔧 You had "pposts.append" here by mistake
                    "title": title,
                    "text": text,
                    "comments": comments
                })

        return pd.DataFrame(posts)

