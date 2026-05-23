import feedparser
from bs4 import BeautifulSoup
import streamlit as st
from typing import List

@st.cache_data(ttl=600)
def fetch_rss_news(feed_urls: List[str], max_articles_per_feed: int = 8):
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles_per_feed]:
                raw_summary = entry.get('summary', '') or entry.get('description', '')
                soup = BeautifulSoup(raw_summary, "html.parser")
                clean_text = soup.get_text(separator=" ", strip=True)
                full_text = f"{entry.title}. {clean_text}"
                articles.append({
                    "id": entry.link,
                    "url": entry.link,
                    "text": full_text,
                    "timestamp": entry.get('published', '')
                })
        except Exception:
            continue
    return articles