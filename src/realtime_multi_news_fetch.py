import os, re, pickle, pandas as pd
from src.build_real_dataset import build

def collect():
    """Fetch fresh news, classify each headline, save results."""
    print("Collecting fresh news...")
    build()

    # Load model
    try:
        model = pickle.load(open("models/fake_news_model.pkl", "rb"))
        vec = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    except FileNotFoundError:
        print("  Model not trained yet — run src/train_fake_news.py first")
        return

    # Load & classify
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))

    def clean(t):
        t = str(t).lower()
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^a-z ]", "", t)
        return " ".join(w for w in t.split() if w not in stop)

    df = pd.read_csv("data/raw/real_news_master.csv")
    df["cleaned"] = df["text"].apply(clean)

    probs = model.predict_proba(vec.transform(df["cleaned"]))
    df["truth_score"] = [round(p[1] * 100, 2) for p in probs]
    df["verdict"] = ["TRUE" if p[1] >= 0.5 else "FAKE" for p in probs]
    df = df.drop(columns=["cleaned"])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/classified_news.csv", index=False)
    print(f"  Classified {len(df)} articles → data/processed/classified_news.csv")