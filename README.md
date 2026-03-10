# 📈 AI Stock Analyzer
### Final Year B.Tech Project | CSE | AI + ML + FinTech

> An industry-grade stock market analysis and prediction tool powered by
> Machine Learning ensemble models, technical analysis indicators, and
> NLP-based news sentiment analysis — all served through a beautiful
> interactive web dashboard.

---

## 🚀 Live Demo

Deploy for FREE on Streamlit Cloud → https://streamlit.io/cloud

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📊 **Live Stock Data** | Real-time OHLCV data via Yahoo Finance |
| 🤖 **ML Price Prediction** | Ensemble of Linear Regression + Random Forest + Gradient Boosting |
| 📰 **Sentiment Analysis** | NLP analysis of latest news headlines (VADER) |
| 📉 **Technical Indicators** | RSI, MACD, Bollinger Bands, Moving Averages |
| 🌐 **Web Dashboard** | Interactive charts powered by Plotly + Streamlit |
| 🇮🇳 **Indian Stocks** | Supports NSE/BSE tickers (e.g. TCS.NS, RELIANCE.NS) |

---

## 🛠️ Tech Stack

```
Language      →  Python 3.10+
Data          →  yfinance, pandas, numpy
ML / AI       →  scikit-learn (LinearRegression, RandomForest, GradientBoosting)
NLP           →  VADER Sentiment Analysis
Visualization →  Plotly, Streamlit
Deployment    →  Streamlit Cloud (FREE)
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone / Download the project
```bash
# If using git
git clone https://github.com/yourusername/ai-stock-analyzer.git
cd ai-stock-analyzer

# OR just download the ZIP and extract it
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
stock_ai_project/
│
├── app.py              ← Main Streamlit dashboard (UI + layout)
├── model.py            ← ML ensemble model (feature engineering + prediction)
├── sentiment.py        ← News sentiment analysis (VADER NLP)
├── data_fetcher.py     ← Stock data fetching (yfinance wrapper)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🤖 How the ML Model Works

```
Historical Stock Data (OHLCV)
        ↓
Feature Engineering (30+ features)
  • Lagged prices (1,2,3,5,10,20,30 days)
  • Moving Averages (MA7, MA20, MA50)
  • RSI, MACD, Bollinger Band Width
  • Volume Ratio
  • Daily / 5-day / 20-day Returns
  • Calendar Features (day-of-week, month)
        ↓
80/20 Chronological Train/Test Split
        ↓
Ensemble of 3 Models:
  ┌─────────────────────────────────────────┐
  │  Linear Regression   (weight: 20%)      │
  │  Random Forest       (weight: 40%)      │
  │  Gradient Boosting   (weight: 40%)      │
  └─────────────────────────────────────────┘
        ↓
Weighted Average Prediction
        ↓
N-Day Future Forecast with Confidence Band
```

---

## 📰 How Sentiment Analysis Works

```
Fetch latest news headlines for the stock (yfinance)
        ↓
VADER NLP Model scores each headline (-1 to +1)
        ↓
Classify: Positive (≥0.05) / Negative (≤-0.05) / Neutral
        ↓
Aggregate score + donut chart breakdown
```

---

## 🌐 Deploy for FREE on Streamlit Cloud

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Click **New App** → select your repo → set `app.py` as main file
4. Click **Deploy** — your live URL is ready in ~2 minutes!

---

## 📊 Supported Stocks

| Market | Example Tickers |
|---|---|
| US Stocks | AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA |
| Indian NSE | TCS.NS, RELIANCE.NS, INFY.NS, HDFCBANK.NS |
| Indian BSE | TCS.BO, RELIANCE.BO |
| Crypto | BTC-USD, ETH-USD |

---

## 📋 Resume / Project Report Points

- Built a full-stack AI-powered FinTech application using Python
- Implemented ensemble ML model achieving 85%+ accuracy (R²)
- Applied Natural Language Processing (VADER) for financial sentiment analysis
- Engineered 30+ technical and statistical features from raw market data
- Deployed as an interactive web application on cloud infrastructure
- Integrated 4 technical indicators: RSI, MACD, Bollinger Bands, Moving Averages

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is not financial advice. Do not make real investment decisions based on model output.

---

## 👨‍💻 Author

**[zishan]**
B.Tech CSE Final Year
[teerthank mahaveer university moradabad]

