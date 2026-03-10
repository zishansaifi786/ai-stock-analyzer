import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from model import StockPredictor
from sentiment import SentimentAnalyzer
from data_fetcher import StockDataFetcher

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e6f0;
}

.main-header {
    background: linear-gradient(90deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-header {
    text-align: center;
    color: #8892a4;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    backdrop-filter: blur(10px);
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 0.2rem;
}

.positive { color: #00e676; }
.negative { color: #ff5252; }
.neutral  { color: #00d4ff; }

.sentiment-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-positive { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid #00e676; }
.badge-negative { background: rgba(255,82,82,0.15);  color: #ff5252; border: 1px solid #ff5252; }
.badge-neutral  { background: rgba(0,212,255,0.12);  color: #00d4ff; border: 1px solid #00d4ff; }

.news-card {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #00d4ff;
    border-bottom: 1px solid rgba(0,212,255,0.2);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

.stButton > button {
    background: linear-gradient(90deg, #00d4ff, #7b2fff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2.5rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95);
    border-right: 1px solid rgba(0,212,255,0.1);
}

.prediction-box {
    background: linear-gradient(135deg, rgba(123,47,255,0.15), rgba(0,212,255,0.1));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">📈 AI Stock Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Machine Learning & Sentiment Analysis</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    ticker = st.text_input("Stock Ticker Symbol", value="AAPL",
                           help="E.g. AAPL, TSLA, GOOGL, MSFT, TCS.NS").upper().strip()

    period_map = {
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    period_label = st.selectbox("Historical Data Period", list(period_map.keys()), index=1)
    period = period_map[period_label]

    forecast_days = st.slider("Forecast Days Ahead", min_value=5, max_value=30, value=15)

    st.markdown("---")
    st.markdown("### 📌 Popular Tickers")
    cols = st.columns(2)
    popular = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "TCS.NS", "RELIANCE.NS"]
    for i, t in enumerate(popular):
        with cols[i % 2]:
            if st.button(t, key=f"btn_{t}"):
                ticker = t

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Stock")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:#8892a4;'>
    ⚠️ For educational purposes only.<br>
    Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if analyze_btn or st.session_state.get("auto_run"):

    with st.spinner(f"🔍 Fetching data for **{ticker}**..."):
        fetcher = StockDataFetcher(ticker, period)
        df, info = fetcher.fetch()

    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for '{ticker}'. Please check the ticker symbol.")
        st.stop()

    # ── Company Info Row ──────────────────────
    st.markdown(f"## {info.get('longName', ticker)}  `{ticker}`")

    c1, c2, c3, c4 = st.columns(4)
    current_price = df['Close'].iloc[-1]
    prev_price    = df['Close'].iloc[-2]
    change        = current_price - prev_price
    change_pct    = (change / prev_price) * 100
    color_cls     = "positive" if change >= 0 else "negative"
    arrow         = "▲" if change >= 0 else "▼"

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value neutral">${current_price:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Change</div>
            <div class="metric-value {color_cls}">{arrow} {change_pct:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        high_52 = df['High'].rolling(252).max().iloc[-1]
        low_52  = df['Low'].rolling(252).min().iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">52-Week High</div>
            <div class="metric-value positive">${high_52:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">52-Week Low</div>
            <div class="metric-value negative">${low_52:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "🤖 ML Prediction", "📰 Sentiment", "📋 Technical Analysis"])

    # ─── TAB 1: PRICE CHART ───────────────────
    with tab1:
        st.markdown('<div class="section-title">Historical Price + Volume</div>', unsafe_allow_html=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Price",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252"
        ), row=1, col=1)

        # Moving Averages
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()

        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20",
                                 line=dict(color="#00d4ff", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="MA 50",
                                 line=dict(color="#7b2fff", width=1.5, dash="dash")), row=1, col=1)

        # Volume bars
        colors = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                             marker_color=colors, opacity=0.6), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#8892a4"),
            xaxis_rangeslider_visible=False,
            height=520,
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    # ─── TAB 2: ML PREDICTION ─────────────────
    with tab2:
        st.markdown('<div class="section-title">Machine Learning Price Prediction</div>', unsafe_allow_html=True)

        with st.spinner("🧠 Training ML model..."):
            predictor = StockPredictor(df, forecast_days=forecast_days)
            predictions, future_dates, metrics = predictor.train_and_predict()

        # Accuracy metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy (R²)</div>
                <div class="metric-value neutral">{metrics['r2']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Abs Error</div>
                <div class="metric-value neutral">${metrics['mae']:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            pred_change = ((predictions[-1] - current_price) / current_price) * 100
            pred_color  = "positive" if pred_change >= 0 else "negative"
            pred_arrow  = "▲" if pred_change >= 0 else "▼"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{forecast_days}-Day Forecast</div>
                <div class="metric-value {pred_color}">{pred_arrow} {pred_change:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        # Prediction box
        final_price = predictions[-1]
        fp_color    = "#00e676" if final_price > current_price else "#ff5252"
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-family:Space Mono,monospace; font-size:0.8rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.1em;">
                Predicted Price in {forecast_days} Days
            </div>
            <div style="font-family:Syne,sans-serif; font-size:3rem; font-weight:800; color:{fp_color}; margin:0.3rem 0;">
                ${final_price:.2f}
            </div>
            <div style="font-size:0.9rem; color:#8892a4;">
                Current: ${current_price:.2f} → Target: ${final_price:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction chart
        fig2 = go.Figure()
        last_n = min(60, len(df))
        hist_dates = df.index[-last_n:]
        hist_prices = df['Close'].values[-last_n:]

        fig2.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            name="Historical", line=dict(color="#00d4ff", width=2)
        ))

        # Connect last historical point to first predicted point
        connect_dates  = [hist_dates[-1]] + list(future_dates)
        connect_prices = [hist_prices[-1]] + list(predictions)

        fig2.add_trace(go.Scatter(
            x=connect_dates, y=connect_prices,
            name="Predicted", line=dict(color="#7b2fff", width=2.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(123,47,255,0.06)"
        ))

        # Confidence band
        std_err = metrics['mae']
        upper = [p + std_err * 1.5 for p in predictions]
        lower = [p - std_err * 1.5 for p in predictions]

        fig2.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(0,212,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band"
        ))

        fig2.add_vline(x=str(df.index[-1]), line_dash="dash",
                       line_color="rgba(255,255,255,0.2)")
        fig2.add_annotation(x=str(df.index[-1]), y=current_price,
                            text="Today", showarrow=True, arrowcolor="#8892a4",
                            font=dict(color="#8892a4", size=11))

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#8892a4"),
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.05)
        )
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

        # Model info
        with st.expander("ℹ️ How the model works"):
            st.markdown("""
            This project uses an **ensemble of ML models**:

            - **Linear Regression** — captures the overall trend
            - **Random Forest** — captures non-linear price patterns
            - **Gradient Boosting** — refines predictions by correcting errors

            **Features used:**
            - Lagged prices (1, 5, 10, 20, 30 days)
            - Moving averages (MA7, MA20, MA50)
            - RSI (Relative Strength Index)
            - MACD (Moving Average Convergence Divergence)
            - Bollinger Bands
            - Volume trends
            - Day of week / month (calendar features)
            """)

    # ─── TAB 3: SENTIMENT ─────────────────────
    with tab3:
        st.markdown('<div class="section-title">News Sentiment Analysis</div>', unsafe_allow_html=True)

        with st.spinner("📰 Analyzing news sentiment..."):
            analyzer = SentimentAnalyzer(ticker)
            sentiment_data = analyzer.analyze()

        overall = sentiment_data['overall']
        score   = sentiment_data['score']

        badge_class = {
            "Positive": "badge-positive",
            "Negative": "badge-negative",
            "Neutral":  "badge-neutral"
        }.get(overall, "badge-neutral")

        s1, s2, s3 = st.columns([1, 1, 2])
        with s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div style="margin-top:0.5rem">
                    <span class="sentiment-badge {badge_class}">{overall}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with s2:
            score_color = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Score</div>
                <div class="metric-value {score_color}">{score:+.3f}</div>
            </div>""", unsafe_allow_html=True)

        with s3:
            pos = sentiment_data['positive_count']
            neg = sentiment_data['negative_count']
            neu = sentiment_data['neutral_count']
            total = pos + neg + neu or 1
            fig_donut = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos, neg, neu],
                hole=0.65,
                marker_colors=["#00e676", "#ff5252", "#00d4ff"],
                textinfo="label+percent"
            ))
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                font=dict(family="Space Mono", color="#8892a4", size=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown('<div class="section-title">Recent News Headlines</div>', unsafe_allow_html=True)
        for article in sentiment_data['articles'][:8]:
            s_class = {
                "Positive": "badge-positive",
                "Negative": "badge-negative",
                "Neutral":  "badge-neutral"
            }.get(article['sentiment'], "badge-neutral")

            st.markdown(f"""
            <div class="news-card">
                <span class="sentiment-badge {s_class}" style="font-size:0.65rem; padding:0.2rem 0.6rem">
                    {article['sentiment']}
                </span>
                <span style="font-size:0.85rem; margin-left:0.6rem; color:#c8d0dc;">
                    {article['headline']}
                </span>
                <div style="font-family:Space Mono,monospace; font-size:0.65rem; color:#8892a4; margin-top:0.3rem;">
                    Score: {article['score']:+.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── TAB 4: TECHNICAL ANALYSIS ────────────
    with tab4:
        st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)

        # RSI
        delta     = df['Close'].diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        rs        = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp12       = df['Close'].ewm(span=12, adjust=False).mean()
        exp26       = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']  = exp12 - exp26
        df['Signal']= df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_mid']   = df['Close'].rolling(20).mean()
        rolling_std    = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + 2 * rolling_std
        df['BB_lower'] = df['BB_mid'] - 2 * rolling_std

        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=("Bollinger Bands", "RSI", "MACD"),
                             row_heights=[0.5, 0.25, 0.25],
                             vertical_spacing=0.05)

        # Bollinger
        fig3.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close",
                                  line=dict(color="#00d4ff", width=1.5)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name="Upper Band",
                                  line=dict(color="#7b2fff", width=1, dash="dot")), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name="Lower Band",
                                  line=dict(color="#7b2fff", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(123,47,255,0.07)"), row=1, col=1)

        # RSI
        rsi_color = ["#00e676" if r < 30 else "#ff5252" if r > 70 else "#00d4ff" for r in df['RSI'].fillna(50)]
        fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
                                  line=dict(color="#00d4ff", width=1.5)), row=2, col=1)
        fig3.add_hline(y=70, line_dash="dash", line_color="#ff5252", opacity=0.5, row=2, col=1)
        fig3.add_hline(y=30, line_dash="dash", line_color="#00e676", opacity=0.5, row=2, col=1)

        # MACD
        macd_colors = ["#00e676" if m >= 0 else "#ff5252" for m in (df['MACD'] - df['Signal']).fillna(0)]
        fig3.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'],
                              name="MACD Hist", marker_color=macd_colors, opacity=0.7), row=3, col=1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
                                  line=dict(color="#00d4ff", width=1.2)), row=3, col=1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal",
                                  line=dict(color="#7b2fff", width=1.2)), row=3, col=1)

        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#8892a4"),
            height=650,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig3, use_container_width=True)

        # Signal Summary
        st.markdown('<div class="section-title">Signal Summary</div>', unsafe_allow_html=True)
        latest_rsi  = df['RSI'].iloc[-1]
        latest_macd = df['MACD'].iloc[-1]
        latest_sig  = df['Signal'].iloc[-1]
        latest_close= df['Close'].iloc[-1]
        latest_bbm  = df['BB_mid'].iloc[-1]

        signals = []
        signals.append(("RSI",
                         "Oversold (BUY)" if latest_rsi < 30 else "Overbought (SELL)" if latest_rsi > 70 else "Neutral",
                         "positive" if latest_rsi < 30 else "negative" if latest_rsi > 70 else "neutral",
                         f"{latest_rsi:.1f}"))
        signals.append(("MACD",
                         "Bullish Cross" if latest_macd > latest_sig else "Bearish Cross",
                         "positive" if latest_macd > latest_sig else "negative",
                         f"{latest_macd:.3f}"))
        signals.append(("Bollinger",
                         "Above Mid (Bullish)" if latest_close > latest_bbm else "Below Mid (Bearish)",
                         "positive" if latest_close > latest_bbm else "negative",
                         f"Mid: ${latest_bbm:.2f}"))
        signals.append(("MA20 vs MA50",
                         "Golden Cross (Bullish)" if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else "Death Cross (Bearish)",
                         "positive" if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else "negative",
                         f"MA20: ${df['MA20'].iloc[-1]:.2f}"))

        sig_cols = st.columns(4)
        for i, (name, signal, color, value) in enumerate(signals):
            with sig_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value {color}" style="font-size:0.95rem; margin:0.4rem 0">{signal}</div>
                    <div style="font-family:Space Mono,monospace; font-size:0.7rem; color:#8892a4">{value}</div>
                </div>""", unsafe_allow_html=True)

    # ── Raw Data ──────────────────────────────
    with st.expander("📄 View Raw Data"):
        display_df = df[['Open','High','Low','Close','Volume']].tail(30).round(2)
        st.dataframe(display_df.style.background_gradient(subset=['Close'], cmap='Blues'),
                     use_container_width=True)

else:
    # ── Welcome screen ────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#8892a4;">
        <div style="font-size:5rem; margin-bottom:1rem;">📊</div>
        <div style="font-family:Syne,sans-serif; font-size:1.5rem; color:#c8d0dc; font-weight:600; margin-bottom:0.5rem;">
            Enter a stock ticker and click Analyze
        </div>
        <div style="font-family:Space Mono,monospace; font-size:0.85rem;">
            Try: AAPL · TSLA · GOOGL · MSFT · TCS.NS · RELIANCE.NS
        </div>
    </div>
    """, unsafe_allow_html=True)
