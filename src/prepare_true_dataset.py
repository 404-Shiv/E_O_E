import pandas as pd

df = pd.read_csv("data/raw/real_news_master.csv")
df = df[["text"]].dropna()
df["label"] = 1

df.to_csv("data/raw/True.csv", index=False)
print("True dataset created:", len(df), "rows")