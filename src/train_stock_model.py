import yfinance as yf
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

symbols = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]
data = []

for s in symbols:
    hist = yf.Ticker(s).history(period="2y")
    hist["ret"] = hist["Close"].pct_change()
    hist["vol"] = hist["ret"].rolling(20).std()
    hist["ma50"] = hist["Close"].rolling(50).mean()
    hist["ma200"] = hist["Close"].rolling(200).mean()
    hist = hist.dropna()

    for _, r in hist.iterrows():
        if r["ret"] > 0.02:
            label = 2
        elif r["ret"] > 0:
            label = 1
        else:
            label = 0
        data.append([r["ret"], r["vol"], r["ma50"], r["ma200"], label])

df = pd.DataFrame(data, columns=["ret", "vol", "ma50", "ma200", "label"])

X = df[["ret", "vol", "ma50", "ma200"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

pickle.dump(model, open("models/stock_model.pkl", "wb"))
print("✅ Stock Model Trained")
