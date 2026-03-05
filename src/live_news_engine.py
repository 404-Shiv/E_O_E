import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FINN_KEY = os.getenv("FINNHUB_API_KEY")
NEWS_KEY = os.getenv("NEWS_API_KEY")


def get_company_news_finnhub(symbol):
    today = datetime.today()
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"

    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": FINN_KEY
    }

    res = requests.get(url, params=params).json()

    news_list = []

    if isinstance(res, list):
        for article in res[:10]:
            news_list.append({
                "title": article.get("headline"),
                "source": article.get("source"),
                "url": article.get("url"),
                "date": article.get("datetime")
            })

    return news_list


def get_company_news_newsapi(company_name):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": company_name,
        "language": "en",
        "pageSize": 10,
        "apiKey": NEWS_KEY
    }

    res = requests.get(url, params=params).json()

    news_list = []

    for article in res.get("articles", []):
        news_list.append({
            "title": article.get("title"),
            "source": article["source"]["name"],
            "url": article.get("url"),
            "date": article.get("publishedAt")
        })

    return news_list


def get_live_company_news(company_input):
    """
    Main function called by frontend
    """

    # Try Finnhub first if input looks like stock symbol
    if company_input.isupper() or "." in company_input:
        news = get_company_news_finnhub(company_input)
        if news:
            return news

    # Fallback to NewsAPI
    return get_company_news_newsapi(company_input)