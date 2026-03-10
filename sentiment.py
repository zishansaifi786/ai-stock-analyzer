"""
sentiment.py
─────────────
News sentiment analysis using VADER (Valence Aware Dictionary and
sEntiment Reasoner) — a rule-based model tuned for financial text.

Falls back to curated simulated headlines when live news is
unavailable (e.g. no internet in a demo environment).
"""

import random
import re
from datetime import datetime, timedelta

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ─────────────────────────────────────────────
#  Simulated news templates (demo / offline)
# ─────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "{ticker} beats quarterly earnings expectations by 12%",
    "{ticker} announces record revenue growth in Q3",
    "Analysts upgrade {ticker} to Strong Buy with $220 target",
    "{ticker} launches groundbreaking AI-powered product line",
    "{ticker} expands into emerging markets, stock surges",
    "{ticker} secures $5B government contract",
    "{ticker} raises full-year guidance on strong demand",
    "Institutional investors increase {ticker} stake by 8%",
]

NEGATIVE_TEMPLATES = [
    "{ticker} misses revenue estimates, shares drop 5%",
    "{ticker} faces antitrust investigation in Europe",
    "{ticker} CEO steps down amid accounting irregularities",
    "Supply chain disruptions impact {ticker} production outlook",
    "{ticker} cuts annual guidance citing macro headwinds",
    "Short sellers target {ticker} citing overvaluation concerns",
    "{ticker} recalls product line due to safety concerns",
]

NEUTRAL_TEMPLATES = [
    "{ticker} to present at upcoming investor conference",
    "{ticker} announces board member election results",
    "{ticker} files 10-Q with SEC for Q3 financials",
    "{ticker} appoints new CFO from within the company",
    "{ticker} to host virtual analyst day next month",
    "Market watches {ticker} ahead of Fed policy decision",
]


class SentimentAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker.replace(".NS", "").replace(".BSE", "")
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None

    # ─────────────────────────────────────────
    #  Classify a single score
    # ─────────────────────────────────────────
    @staticmethod
    def _classify(score: float) -> str:
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        return "Neutral"

    # ─────────────────────────────────────────
    #  Score a headline
    # ─────────────────────────────────────────
    def _score(self, headline: str) -> float:
        if self.vader:
            return self.vader.polarity_scores(headline)['compound']
        # Lightweight keyword fallback
        pos_words = ["beat", "record", "growth", "upgrade", "strong",
                     "buy", "surges", "raises", "expand", "breakthrough"]
        neg_words = ["miss", "drop", "cut", "recall", "investigation",
                     "headwind", "loss", "decline", "concern", "short"]
        h = headline.lower()
        score = sum(0.15 for w in pos_words if w in h) \
              - sum(0.15 for w in neg_words if w in h)
        return max(-1.0, min(1.0, score))

    # ─────────────────────────────────────────
    #  Try to fetch real news from yfinance
    # ─────────────────────────────────────────
    def _fetch_real_news(self):
        if not YF_AVAILABLE:
            return []
        try:
            stock   = yf.Ticker(self.ticker)
            news    = stock.news or []
            results = []
            for item in news[:15]:
                title = item.get('title', '')
                if title:
                    results.append(title)
            return results
        except Exception:
            return []

    # ─────────────────────────────────────────
    #  Build simulated news (for offline use)
    # ─────────────────────────────────────────
    def _simulated_news(self):
        random.seed(hash(self.ticker) % 9999)
        headlines = []

        n_pos = random.randint(4, 6)
        n_neg = random.randint(2, 4)
        n_neu = random.randint(3, 5)

        for _ in range(n_pos):
            t = random.choice(POSITIVE_TEMPLATES).format(ticker=self.ticker)
            headlines.append(t)
        for _ in range(n_neg):
            t = random.choice(NEGATIVE_TEMPLATES).format(ticker=self.ticker)
            headlines.append(t)
        for _ in range(n_neu):
            t = random.choice(NEUTRAL_TEMPLATES).format(ticker=self.ticker)
            headlines.append(t)

        random.shuffle(headlines)
        return headlines

    # ─────────────────────────────────────────
    #  Main analysis method
    # ─────────────────────────────────────────
    def analyze(self) -> dict:
        headlines = self._fetch_real_news()
        source    = "live"
        if not headlines:
            headlines = self._simulated_news()
            source    = "simulated"

        articles = []
        for h in headlines:
            score = self._score(h)
            articles.append({
                'headline' : h,
                'score'    : score,
                'sentiment': self._classify(score)
            })

        # Sort by absolute score (most opinionated first)
        articles.sort(key=lambda x: abs(x['score']), reverse=True)

        scores     = [a['score'] for a in articles]
        avg_score  = sum(scores) / len(scores) if scores else 0.0
        overall    = self._classify(avg_score)

        pos = sum(1 for a in articles if a['sentiment'] == "Positive")
        neg = sum(1 for a in articles if a['sentiment'] == "Negative")
        neu = sum(1 for a in articles if a['sentiment'] == "Neutral")

        return {
            'overall'        : overall,
            'score'          : avg_score,
            'articles'       : articles,
            'positive_count' : pos,
            'negative_count' : neg,
            'neutral_count'  : neu,
            'source'         : source,
        }
