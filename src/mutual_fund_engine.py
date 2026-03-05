import os
import requests
from dotenv import load_dotenv

load_dotenv()

FMP_KEY = os.getenv("FMP_API_KEY")
TWELVE_KEY = os.getenv("TWELVE_API_KEY")

def get_fund_profile(symbol):
    """
    Fetch mutual fund details from FinancialModelingPrep
    """
    url = f"https://financialmodelingprep.com/api/v3/mutual-fund/{symbol}?apikey={FMP_KEY}"
    res = requests.get(url).json()

    if not res:
        return {"error": "Fund not found"}

    data = res[0]

    return {
        "Fund Name": data.get("name"),
        "NAV": data.get("nav"),
        "Expense Ratio": data.get("expenseRatio"),
        "Fund Family": data.get("fundFamily"),
        "Category": data.get("category"),
        "Total Assets (AUM)": data.get("totalAssets")
    }


def get_live_price(symbol):
    """
    Fetch real-time price from Twelve Data
    """
    url = "https://api.twelvedata.com/price"
    params = {
        "symbol": symbol,
        "apikey": TWELVE_KEY
    }

    res = requests.get(url, params=params).json()

    return {
        "Live Price": res.get("price"),
        "Status": res.get("status")
    }


def get_rsi(symbol):
    """
    Fetch RSI indicator from Twelve Data
    """
    url = "https://api.twelvedata.com/rsi"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "time_period": 14,
        "apikey": TWELVE_KEY
    }

    res = requests.get(url, params=params).json()

    try:
        rsi_value = res["values"][0]["rsi"]
        return {"RSI (14)": rsi_value}
    except:
        return {"RSI (14)": "Not Available"}


def full_mutual_fund_analysis(symbol):
    profile = get_fund_profile(symbol)
    price = get_live_price(symbol)
    rsi = get_rsi(symbol)

    return {**profile, **price, **rsi}