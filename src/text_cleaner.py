import re

def clean_text(text):
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()
