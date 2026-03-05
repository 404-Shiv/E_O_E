import pandas as pd, re, pickle, nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")
stop=set(stopwords.words("english"))

true=pd.read_csv("data/raw/True.csv")
fake=pd.read_csv("data/raw/Fake.csv")

df=pd.concat([true,fake]).sample(frac=1)

def clean(t):
    t=t.lower()
    t=re.sub(r"http\S+","",t)
    t=re.sub(r"[^a-z ]","",t)
    return " ".join(w for w in t.split() if w not in stop)

df["text"]=df["text"].apply(clean)

vec=TfidfVectorizer(max_features=5000)
X=vec.fit_transform(df["text"])

model=LogisticRegression(max_iter=1000)
model.fit(X,df["label"])

pickle.dump(model,open("models/fake_news_model.pkl","wb"))
pickle.dump(vec,open("models/tfidf_vectorizer.pkl","wb"))

print("Fake news model trained")