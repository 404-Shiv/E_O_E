import feedparser
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords

# -------------------------------------------------
# Load trained fake-news ML model
# -------------------------------------------------
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

# -------------------------------------------------
# HARD-CODED RSS SOURCES (INDIA + GLOBAL)
# -------------------------------------------------
RSS_FEEDS = {
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "The Hindu BusinessLine": "https://www.thehindubusinessline.com/markets/feeder/default.rss"
}

true_news = []
false_news = []

# -------------------------------------------------
# Fetch + classify news
# -------------------------------------------------
for source, url in RSS_FEEDS.items():
    feed = feedparser.parse(url)

    for entry in feed.entries[:15]:
        headline = entry.title
        cleaned = clean_text(headline)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        record = {
            "title": headline,
            "source": source
        }

        if prediction == 1:
            true_news.append(record)
        else:
            false_news.append(record)

# -------------------------------------------------
# Save results
# -------------------------------------------------
pd.DataFrame(true_news).to_csv(
    "data/processed/live_true_news.csv", index=False
)
pd.DataFrame(false_news).to_csv(
    "data/processed/live_false_news.csv", index=False
)

print("[OK] RSS Fetch Completed")
print(f"TRUE news count  : {len(true_news)}")
print(f"FALSE news count : {len(false_news)}")
