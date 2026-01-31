import pandas as pd
import re, pickle, nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

true_df = pd.read_csv("data/raw/True.csv")
fake_df = pd.read_csv("data/raw/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

pickle.dump(model, open("models/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))

print("✅ Fake News Model Trained")
