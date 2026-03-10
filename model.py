"""
model.py
─────────
Ensemble ML model for stock price prediction.

Models used:
  • Linear Regression   – baseline trend
  • Random Forest       – non-linear patterns
  • Gradient Boosting   – error correction

Features engineered:
  • Lagged close prices
  • Rolling averages (MA7, MA20, MA50)
  • RSI, MACD, Bollinger Band width
  • Volume trend
  • Calendar features (day-of-week, month)
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


class StockPredictor:
    def __init__(self, df: pd.DataFrame, forecast_days: int = 15):
        self.df            = df.copy()
        self.forecast_days = forecast_days
        self.scaler        = StandardScaler()

    # ─────────────────────────────────────────
    #  FEATURE ENGINEERING
    # ─────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # Lagged prices
        for lag in [1, 2, 3, 5, 10, 20, 30]:
            d[f'lag_{lag}'] = d['Close'].shift(lag)

        # Moving averages
        for w in [7, 20, 50]:
            d[f'ma_{w}'] = d['Close'].rolling(w).mean()

        # MA ratios (relative strength vs trend)
        d['ma7_ma20_ratio']  = d['ma_7']  / d['ma_20']
        d['ma20_ma50_ratio'] = d['ma_20'] / d['ma_50']

        # Volatility
        d['rolling_std_20']  = d['Close'].rolling(20).std()
        d['bb_width']        = d['rolling_std_20'] / d['ma_20']

        # RSI (14-day)
        delta  = d['Close'].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-9)
        d['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp12      = d['Close'].ewm(span=12, adjust=False).mean()
        exp26      = d['Close'].ewm(span=26, adjust=False).mean()
        d['macd']  = exp12 - exp26
        d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
        d['macd_diff']   = d['macd'] - d['macd_signal']

        # Volume features
        d['volume_ma20']   = d['Volume'].rolling(20).mean()
        d['volume_ratio']  = d['Volume'] / (d['volume_ma20'] + 1)

        # Daily return
        d['daily_return']  = d['Close'].pct_change()
        d['return_5d']     = d['Close'].pct_change(5)
        d['return_20d']    = d['Close'].pct_change(20)

        # Calendar
        d['day_of_week']   = d.index.dayofweek
        d['month']         = d.index.month
        d['quarter']       = d.index.quarter

        return d

    # ─────────────────────────────────────────
    #  TRAIN & PREDICT
    # ─────────────────────────────────────────
    def train_and_predict(self):
        df = self._build_features(self.df).dropna()

        feature_cols = [c for c in df.columns
                        if c not in ('Open', 'High', 'Low', 'Close',
                                     'Volume', 'Dividends', 'Stock Splits')]

        X = df[feature_cols].values
        y = df['Close'].values

        # Train / test split (chronological – no shuffle)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # ── Three models ──
        lr  = LinearRegression()
        rf  = RandomForestRegressor(n_estimators=150, max_depth=8,
                                     random_state=42, n_jobs=-1)
        gb  = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                         learning_rate=0.05, random_state=42)

        lr.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)
        gb.fit(X_train_s, y_train)

        # Ensemble prediction (weighted average)
        w_lr, w_rf, w_gb = 0.20, 0.40, 0.40
        y_pred = (w_lr * lr.predict(X_test_s)
                + w_rf * rf.predict(X_test_s)
                + w_gb * gb.predict(X_test_s))

        metrics = {
            'r2' : max(0.0, r2_score(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

        # ── Future forecast ──
        last_row = df[feature_cols].iloc[-1].values.copy()
        predictions  = []
        future_dates = []
        last_date    = df.index[-1]
        last_close   = float(df['Close'].iloc[-1])

        for i in range(self.forecast_days):
            x_scaled     = self.scaler.transform(last_row.reshape(1, -1))
            pred_ensemble= (w_lr * lr.predict(x_scaled)[0]
                          + w_rf * rf.predict(x_scaled)[0]
                          + w_gb * gb.predict(x_scaled)[0])

            # Add gentle mean-reversion noise
            noise         = np.random.normal(0, metrics['mae'] * 0.10)
            pred_price    = max(pred_ensemble + noise, 0.01)

            predictions.append(pred_price)
            next_date = last_date + timedelta(days=1)
            # Skip weekends
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            future_dates.append(next_date)
            last_date = next_date

            # Roll the lag features forward
            for col_i, col in enumerate(feature_cols):
                if col == 'lag_1':
                    last_row[col_i] = pred_price
                elif col.startswith('lag_'):
                    lag_val = int(col.split('_')[1])
                    if lag_val > 1:
                        shift_col = f'lag_{lag_val - 1}'
                        if shift_col in feature_cols:
                            prev_i = feature_cols.index(shift_col)
                            last_row[col_i] = last_row[prev_i]

        return predictions, future_dates, metrics
