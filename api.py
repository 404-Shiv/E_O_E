"""
EOE Financial Intelligence — Flask API Backend
Serves all engine functions as REST endpoints for the React frontend.
"""
import os, pickle, re
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from src.stock_engine import stock_decision, search_stocks
from src.mf_engine import mutual_fund_decision, search_mutual_funds, get_mf_nav_history
from src.live_news_engine import get_live_company_news
import pandas as pd

app = Flask(__name__)
CORS(app)

# ─── Load fake-news model ───
try:
    model = pickle.load(open("models/fake_news_model.pkl", "rb"))
    vec = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

stop = set(stopwords.words("english"))

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(w for w in t.split() if w not in stop)

# In-memory MF cache
mf_cache = {"df": None}


# ─── News Verifier ───
@app.route("/api/verify", methods=["POST"])
def verify_news():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    cleaned = clean(text)
    probs = model.predict_proba(vec.transform([cleaned]))[0]
    truth_pct = round(probs[1] * 100, 2)
    fake_pct = round(probs[0] * 100, 2)
    return jsonify({
        "truth_pct": truth_pct,
        "fake_pct": fake_pct,
        "verdict": "TRUE" if truth_pct >= 50 else "FAKE",
    })


# ─── Live News Feed ───
@app.route("/api/news")
def get_news():
    classified_path = "data/processed/classified_news.csv"
    raw_path = "data/raw/real_news_master.csv"
    df = None

    if os.path.exists(classified_path):
        df = pd.read_csv(classified_path)
    elif os.path.exists(raw_path) and model_loaded:
        df = pd.read_csv(raw_path)
        df["cleaned"] = df["text"].apply(clean)
        probs = model.predict_proba(vec.transform(df["cleaned"]))
        df["truth_score"] = [round(p[1] * 100, 2) for p in probs]
        df["verdict"] = ["TRUE" if p[1] >= 0.5 else "FAKE" for p in probs]
    else:
        return jsonify({"articles": [], "message": "No news data found"})

    if df is not None and not df.empty:
        if "truth_score" not in df.columns and model_loaded:
            df["cleaned"] = df["text"].apply(clean)
            probs = model.predict_proba(vec.transform(df["cleaned"]))
            df["truth_score"] = [round(p[1] * 100, 2) for p in probs]
            df["verdict"] = ["TRUE" if p[1] >= 0.5 else "FAKE" for p in probs]

        articles = []
        for _, row in df.head(50).iterrows():
            url_val = row.get("url", "")
            url_str = "" if pd.isna(url_val) else str(url_val)
            # Extra safeguard against string "nan" or "NaN"
            if url_str.lower() == "nan":
                url_str = ""
            
            articles.append({
                "text": str(row.get("text", "")),
                "source": str(row.get("source", "Unknown")),
                "date": str(row.get("date", "")),
                "verdict": str(row.get("verdict", "")),
                "truth_score": float(row.get("truth_score", 0)),
                "url": url_str,
            })
        total = len(df)
        true_count = int((df["verdict"] == "TRUE").sum()) if "verdict" in df.columns else 0
        fake_count = int((df["verdict"] == "FAKE").sum()) if "verdict" in df.columns else 0
        return jsonify({
            "articles": articles,
            "total": total,
            "true_count": true_count,
            "fake_count": fake_count,
        })
    return jsonify({"articles": []})


@app.route("/api/news/refresh", methods=["POST"])
def refresh_news():
    try:
        from src.realtime_multi_news_fetch import collect
        collect()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Stock Search & Analysis ───
@app.route("/api/stocks/search")
def stock_search():
    q = request.args.get("q", "")
    results = search_stocks(q)
    return jsonify(results)


@app.route("/api/stocks/analyze/<symbol>")
def stock_analyze(symbol):
    period = request.args.get("period", "3mo")
    result = stock_decision(symbol, period=period)
    if "error" in result:
        return jsonify(result), 400
    # Convert history DF to JSON-safe format with OHLC data
    hist = result.pop("history", None)
    history_data = []
    if hist is not None and not hist.empty:
        for idx, row in hist.iterrows():
            history_data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })
    result["history"] = history_data
    return jsonify(result)


@app.route("/api/stocks/news/<symbol>")
def stock_news(symbol):
    try:
        news = get_live_company_news(symbol)
        return jsonify(news if news else [])
    except Exception:
        return jsonify([])


@app.route("/api/stocks/price/<symbol>")
def stock_price(symbol):
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        try:
            price = ticker.fast_info["lastPrice"]
        except Exception:
            history = ticker.history(period="1d")
            price = history["Close"].iloc[-1] if not history.empty else 0
        return jsonify({"price": round(float(price), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Mutual Funds ───
@app.route("/api/mf/load", methods=["POST"])
def mf_load():
    df = mutual_fund_decision()
    if "error" in df.columns:
        return jsonify({"error": df["error"].iloc[0]}), 500
    mf_cache["df"] = df
    return jsonify({"count": len(df)})


@app.route("/api/mf/search")
def mf_search():
    q = request.args.get("q", "")
    df = mf_cache.get("df")
    if df is None:
        return jsonify({"error": "Load fund data first"}), 400
    filtered = search_mutual_funds(q, df)
    records = []
    for _, row in filtered.iterrows():
        records.append({
            "code": str(row["code"]),
            "scheme": str(row["scheme"]),
            "nav": float(row["nav"]),
            "date": str(row["date"]),
            "category": str(row["category"]),
        })
    return jsonify(records)


@app.route("/api/mf/analyze/<code>")
def mf_analyze(code):
    hist_df, metrics = get_mf_nav_history(code)
    if "error" in metrics:
        return jsonify(metrics), 400
    history_data = []
    if hist_df is not None:
        for _, row in hist_df.iterrows():
            history_data.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "nav": round(row["nav"], 4),
            })
    return jsonify({
        "metrics": metrics,
        "history": history_data
    })


@app.route("/api/mf/price/<code>")
def mf_price(code):
    try:
        df = mf_cache.get("df")
        if df is not None:
            # Find in cache if loaded
            row = df[df["code"] == str(code)]
            if not row.empty:
                return jsonify({"nav": float(row["nav"].iloc[0])})
        
        # Fallback to direct fetch
        import requests
        url = f"https://api.mfapi.in/mf/{code}"
        resp = requests.get(url, timeout=10).json()
        nav_data = resp.get("data", [])
        if nav_data:
            return jsonify({"nav": float(nav_data[0]["nav"])})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


import time
ticker_cache = {"data": [], "last_fetched": 0}

@app.route("/api/market/ticker")
def market_ticker():
    global ticker_cache
    if time.time() - ticker_cache["last_fetched"] < 15 and ticker_cache["data"]:
        return jsonify(ticker_cache["data"])
    
    symbols = {
        "S&P 500": "^GSPC",
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "DOW JONES": "^DJI",
        "NIKKEI": "^N225",
        "BANK NIFTY": "^NSEBANK",
        "NASDAQ": "^IXIC",
    }
    
    try:
        import yfinance as yf
        import pandas as pd
        data = yf.download(list(symbols.values()), period="5d", progress=False)
        
        results = []
        for name, sym in symbols.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"][sym].dropna()
                else:
                    closes = data["Close"].dropna()
                
                if len(closes) >= 2:
                    prev_close = float(closes.iloc[-2])
                    last = float(closes.iloc[-1])
                elif len(closes) == 1:
                    last = prev_close = float(closes.iloc[-1])
                else:
                    continue
                
                change = last - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0
                
                val_str = f"{last:,.2f}"
                sign = "+" if change >= 0 else "−"
                chg_str = f"{sign}{abs(change):,.2f} ({sign}{abs(change_pct):.2f}%)"
                
                results.append({
                    "name": name,
                    "value": val_str,
                    "change": chg_str,
                    "up": change >= 0
                })
            except Exception:
                pass
                
        if results:
            ticker_cache["data"] = results
            ticker_cache["last_fetched"] = time.time()
            return jsonify(results)
        
        return jsonify(ticker_cache["data"])
    except Exception as e:
        if ticker_cache["data"]:
            return jsonify(ticker_cache["data"])
        return jsonify([]), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
