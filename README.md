#  EOE – Eye on Economy

**EOE (Eye on Economy)** is a full-stack financial intelligence platform that integrates **real-time financial news, AI-based fake news detection, stock market analysis, and mutual fund insights** into a single web application.

The platform helps users analyze Indian stocks, evaluate mutual funds, track global markets, and verify the authenticity of financial news using machine learning.
---

#  Features

### Real-Time Financial News

* Aggregates news from **multiple global financial sources**
* Deduplicates and classifies articles automatically
* News refreshes every **5 minutes**
* Displays **truth scores and source links**

###  AI Fake News Detection

* Detects fake financial news using **Machine Learning**
* Model built with **TF-IDF Vectorizer + Logistic Regression**
* Provides **truth probability and verdict**

###  Stock Analysis

* Search Indian stocks using **keyword matching**
* Technical indicators:

  * SMA-20
  * SMA-50
* Displays:

  * OHLC price history
  * Volume data
  * Price change percentage
* Generates **BUY / HOLD / SELL signals**

###  Mutual Fund Intelligence

* Covers **10,000+ Indian mutual fund schemes**
* Calculates:

  * Returns (1M, 3M, 6M, 1Y)
  * Volatility
  * Maximum drawdown
* Generates **performance verdicts**

###  Live Market Ticker

Tracks global indices including:

* S&P 500
* NIFTY 50
* SENSEX
* NASDAQ

Refresh interval: **15 seconds**

###  Modern User Interface

* Dark / Light theme toggle
* Smooth animations with **Framer Motion**
* Interactive charts using **Recharts**

---

#  Machine Learning Model

**Algorithm:** Logistic Regression
**Feature Extraction:** TF-IDF Vectorizer (Top 5000 features)

### Text Preprocessing

* Lowercasing
* URL removal
* Special character removal
* Stopword filtering using **NLTK**

### Training Dataset

```
data/raw/True.csv
data/raw/Fake.csv
```

### Saved Models

```
models/fake_news_model.pkl
models/tfidf_vectorizer.pkl
```

The model outputs:

* Truth probability
* Fake probability
* Final verdict (TRUE / FAKE)

---

#  Tech Stack

| Layer            | Technologies                                     |
| ---------------- | ------------------------------------------------ |
| Frontend         | React 19, Vite 7, Framer Motion, Recharts, Axios |
| Backend          | Python, Flask, Flask-CORS                        |
| Machine Learning | scikit-learn, TF-IDF, Logistic Regression        |
| NLP              | NLTK                                             |
| Market Data      | yfinance                                         |
| Financial APIs   | NewsAPI, Finnhub, EODHD                          |
| Mutual Fund Data | AMFI India                                       |
| Storage          | CSV datasets, Pickle models                      |

---

#  Data Sources

* NewsAPI
* Finnhub
* EODHD
* Google News RSS
* GDELT
* The Guardian
* yfinance
* AMFI India

---

#  Project Structure

```
eoe_project_final
│
├── api.py
├── app.py
├── requirements.txt
├── .env
│
├── models
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data
│   ├── raw
│   │   ├── True.csv
│   │   └── Fake.csv
│   └── processed
│
├── src
│   ├── stock_engine.py
│   ├── mf_engine.py
│   ├── live_news_engine.py
│   ├── build_real_dataset.py
│   ├── realtime_multi_news_fetch.py
│   └── train_fake_news.py
│
└── frontend
    ├── package.json
    └── src
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        └── components
            ├── Header.jsx
            ├── Ticker.jsx
            ├── LiveNews.jsx
            ├── StockAnalysis.jsx
            ├── StockComparison.jsx
            ├── MutualFunds.jsx
            └── NewsVerifier.jsx
```

---


---

# Running the Project

##  Start Backend

```bash
pip install -r requirements.txt
python api.py
```

Backend runs at:

```
http://localhost:5000
```

---

##  Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:3000
```

---

#  Highlights

* Aggregates **6 financial news sources**
* **AI-powered fake news detection**
* **Technical stock analysis with signals**
* Analysis of **10,000+ mutual funds**
* **Real-time market ticker**
* **Modern React dashboard**

---

<img width="1879" height="994" alt="Screenshot 2026-03-05 141925" src="https://github.com/user-attachments/assets/861738a7-b1db-43c3-bdd1-b80eb8addab1" />
<img width="1866" height="1015" alt="Screenshot 2026-03-05 141738" src="https://github.com/user-attachments/assets/88d85d23-79fa-4eac-9891-3361e3f8cc43" />
<img width="1877" height="1017" alt="Screenshot 2026-03-05 141957" src="https://github.com/user-attachments/assets/0471850e-91a3-4a1e-b243-4ca6fe837627" />
<img width="1906" height="1015" alt="Screenshot 2026-03-05 141831" src="https://github.com/user-attachments/assets/be8f689f-1956-4a5a-ab60-2277ac83bac6" />
<img width="1888" height="742" alt="Screenshot 2026-03-05 141908" src="https://github.com/user-attachments/assets/03fe2bc2-1d8b-486c-a3a1-32556bf848ab" />

.
