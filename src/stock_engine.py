import yfinance as yf
import pickle

model = pickle.load(open("models/stock_model.pkl", "rb"))

def stock_decision(symbol):
    hist = yf.Ticker(symbol).history(period="2y")
    hist["ret"] = hist["Close"].pct_change()
    hist["vol"] = hist["ret"].rolling(20).std()
    hist["ma50"] = hist["Close"].rolling(50).mean()
    hist["ma200"] = hist["Close"].rolling(200).mean()

    hist = hist.dropna().iloc[-1]
    pred = model.predict([[hist["ret"], hist["vol"], hist["ma50"], hist["ma200"]]])[0]

    if pred == 2:
        return "GOOD"
    elif pred == 1:
        return "NEUTRAL"
    else:
        return "UNDERPERFORMING"
