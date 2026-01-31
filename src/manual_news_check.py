import pickle
import re
from nltk.corpus import stopwords

# Load trained model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

# -------------------------------------------------
# MANUAL INPUT
# -------------------------------------------------
news = input("\n📰 Enter financial news headline:\n> ")

cleaned = clean_text(news)
vector = vectorizer.transform([cleaned])

# Get probability scores
probs = model.predict_proba(vector)[0]
false_prob = probs[0] * 100
true_prob = probs[1] * 100

# Final decision
prediction = 1 if true_prob >= false_prob else 0

print("\n📊 RESULT")
print("--------------------------------")

if prediction == 1:
    print("✅ Verdict : TRUE")
else:
    print("❌ Verdict : FAKE")

print(f"Truth Probability : {true_prob:.2f}%")
print(f"Fake Probability  : {false_prob:.2f}%")
