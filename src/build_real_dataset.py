import os, requests, pandas as pd
from dotenv import load_dotenv

load_dotenv()

NEWS = os.getenv("NEWS_API_KEY")
EOD = os.getenv("EODHD_API_KEY")
FINN = os.getenv("FINNHUB_API_KEY")
FILE = "data/raw/real_news_master.csv"


def newsapi():
    """Fetch from NewsAPI (key in .env)."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "india stock market OR RBI OR economy OR mutual fund",
            "language": "en",
            "pageSize": 100,
            "apiKey": NEWS,
        }
        return requests.get(url, params=params, timeout=15).json().get("articles", [])
    except Exception as e:
        print(f"  NewsAPI error: {e}")
        return []


def eodhd():
    """Fetch from EODHD (key in .env)."""
    try:
        url = f"https://eodhd.com/api/news?api_token={EOD}&limit=100"
        data = requests.get(url, timeout=15).json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  EODHD error: {e}")
        return []


def finnhub():
    """Fetch from Finnhub (key in .env)."""
    try:
        url = f"https://finnhub.io/api/v1/news?category=general&token={FINN}"
        data = requests.get(url, timeout=15).json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  Finnhub error: {e}")
        return []


def google_news_rss():
    """Fetch from Google News RSS — FREE, no key needed."""
    try:
        import feedparser
        feed = feedparser.parse(
            "https://news.google.com/rss/search?q=india+finance+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        )
        return [
            {"title": e.get("title", ""), "source": "Google News", "published": e.get("published", ""), "link": e.get("link", "")}
            for e in feed.entries[:80]
        ]
    except Exception as e:
        print(f"  Google News RSS error: {e}")
        return []


def gdelt():
    """Fetch from GDELT API — FREE, no key needed."""
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc?query=india%20finance&mode=artlist&maxrecords=100&format=json"
        data = requests.get(url, timeout=15).json()
        return data.get("articles", [])
    except Exception as e:
        print(f"  GDELT error: {e}")
        return []


def guardian():
    """Fetch from The Guardian Open Platform — FREE with test key."""
    try:
        url = "https://content.guardianapis.com/search"
        params = {
            "q": "india finance stock market",
            "page-size": 50,
            "api-key": "test",  # Guardian provides a free 'test' key
        }
        data = requests.get(url, params=params, timeout=15).json()
        return data.get("response", {}).get("results", [])
    except Exception as e:
        print(f"  Guardian error: {e}")
        return []


def build():
    """Collect from all sources and merge into master CSV."""
    rows = []

    # --- NewsAPI ---
    print("  Fetching NewsAPI...")
    for a in newsapi():
        rows.append({
            "text": a.get("title", ""),
            "source": a.get("source", {}).get("name", "NewsAPI"),
            "date": a.get("publishedAt", ""),
            "url": a.get("url", ""),
        })

    # --- EODHD ---
    print("  Fetching EODHD...")
    for a in eodhd():
        rows.append({
            "text": a.get("title", ""),
            "source": a.get("source", "EODHD"),
            "date": a.get("date", ""),
            "url": a.get("link", ""),
        })

    # --- Finnhub ---
    print("  Fetching Finnhub...")
    for a in finnhub():
        rows.append({
            "text": a.get("headline", ""),
            "source": "Finnhub",
            "date": str(a.get("datetime", "")),
            "url": a.get("url", ""),
        })

    # --- Google News RSS ---
    print("  Fetching Google News RSS...")
    for a in google_news_rss():
        rows.append({
            "text": a.get("title", ""),
            "source": "Google News",
            "date": a.get("published", ""),
            "url": a.get("link", ""),
        })

    # --- GDELT ---
    print("  Fetching GDELT...")
    for a in gdelt():
        rows.append({
            "text": a.get("title", ""),
            "source": a.get("domain", "GDELT"),
            "date": a.get("seendate", ""),
            "url": a.get("url", ""),
        })

    # --- Guardian ---
    print("  Fetching The Guardian...")
    for a in guardian():
        rows.append({
            "text": a.get("webTitle", ""),
            "source": "The Guardian",
            "date": a.get("webPublicationDate", ""),
            "url": a.get("webUrl", ""),
        })

    df = pd.DataFrame(rows)
    df = df[df["text"].str.strip().astype(bool)]  # remove blanks
    df = df.drop_duplicates(subset=["text"])

    # Merge with existing
    try:
        old = pd.read_csv(FILE)
        df = pd.concat([old, df]).drop_duplicates(subset=["text"])
    except Exception:
        pass

    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    df.to_csv(FILE, index=False)
    print(f"  Real dataset size: {len(df)}")


if __name__ == "__main__":
    build()