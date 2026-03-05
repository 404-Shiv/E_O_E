import threading, time, pandas as pd
from src.realtime_multi_news_fetch import collect
from src.stock_engine import stock_decision
from src.mf_engine import mutual_fund_decision

def updater():
    while True:
        try:
            collect()
        except Exception as e:
            print("Update error:", e)
        time.sleep(300)

threading.Thread(target=updater, daemon=True).start()

while True:
    print("\n1 Live News\n2 Stock\n3 Mutual Fund\n4 Exit")
    c = input("Choice: ")

    if c == "1":
        try:
            df = pd.read_csv("data/processed/classified_news.csv")
            print(df[["text", "verdict", "truth_score"]].head(20).to_string())
        except FileNotFoundError:
            print("No classified news yet — wait for background update")
    elif c == "2":
        s = input("Symbol: ")
        print(stock_decision(s))
    elif c == "3":
        print(mutual_fund_decision())
    elif c == "4":
        break