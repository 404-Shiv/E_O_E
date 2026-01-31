import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from src.stock_engine import stock_decision
from src.mf_engine import mutual_fund_decision

# Load ML models
news_model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

def manual_news_check():
    while True:
        news = input("\n📰 Enter financial news (or type 'back' to menu):\n> ")
        if news.lower() == "back":
            break

        vec = vectorizer.transform([clean_text(news)])
        probs = news_model.predict_proba(vec)[0]

        true_prob = probs[1] * 100
        fake_prob = probs[0] * 100

        verdict = "TRUE" if true_prob >= fake_prob else "FAKE"

        print("\n📊 RESULT")
        print("--------------------------------")
        print(f"Verdict          : {verdict}")
        print(f"Truth Probability: {true_prob:.2f}%")
        print(f"Fake Probability : {fake_prob:.2f}%")

def show_live_news():
    print("\n🟢 TRUE NEWS")
    print(pd.read_csv("data/processed/live_true_news.csv").head())

    print("\n🔴 FALSE NEWS")
    print(pd.read_csv("data/processed/live_false_news.csv").head())

def main():
    while True:
        print("\n==============================")
        print(" FINANCIAL NEWS VERIFICATION ")
        print("==============================")
        print("1. Check news manually (TRUE / FALSE)")
        print("2. Show live RSS news result")
        print("3. Check stock performance")
        print("4. Show mutual fund data")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            manual_news_check()
        elif choice == "2":
            show_live_news()
        elif choice == "3":
            symbol = input("Enter stock symbol (e.g., INFY.NS): ")
            print("Decision:", stock_decision(symbol))
        elif choice == "4":
            print(mutual_fund_decision())
        elif choice == "5":
            print("Exiting... Thank you!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
