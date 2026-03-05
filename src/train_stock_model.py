import yfinance as yf
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

symbols = ["INFY.NS","TCS.NS","RELIANCE.NS"]

rows = []

for s in symbols:
    h = yf.Ticker(s).history(period="2y")
    h["ret"] = h["Close"].pct_change()
    h["vol"] = h["ret"].rolling(20).std()
    h["ma50"] = h["Close"].rolling(50).mean()
    h["ma200"] = h["Close"].rolling(200).mean()
    h = h.dropna()

    for _,r in h.iterrows():
        label = 2 if r["ret"] > 0.02 else 1 if r["ret"] > 0 else 0
        rows.append([r["ret"],r["vol"],r["ma50"],r["ma200"],label])

df = pd.DataFrame(rows,columns=["ret","vol","ma50","ma200","label"])

model = RandomForestClassifier()
model.fit(df[["ret","vol","ma50","ma200"]], df["label"])

pickle.dump(model, open("models/stock_model.pkl","wb"))
print("Stock model trained")