# file: analysis/trader_sentiment_analysis_v2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. LOAD DATA
# -----------------------
fear_greed = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")

# -----------------------
# 2. RENAME COLUMNS (IMPORTANT)
# -----------------------

trades = trades.rename(columns={
    'Account': 'account',
    'Coin': 'symbol',
    'Execution Price': 'execution_price',
    'Size Tokens': 'size_tokens',
    'Size USD': 'size_usd',
    'Side': 'side',
    'Timestamp': 'timestamp',
    'Closed PnL': 'closed_pnl'
})

fear_greed = fear_greed.rename(columns={
    'classification': 'Classification',
    'date': 'date'
})

# -----------------------
# 3. DATE HANDLING
# -----------------------

# Fear & Greed (seconds → datetime)
fear_greed['timestamp'] = pd.to_numeric(fear_greed['timestamp'], errors='coerce')
fear_greed['date'] = pd.to_datetime(fear_greed['timestamp'], unit='s')

# Trades (milliseconds → datetime)
trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
trades['date'] = pd.to_datetime(trades['timestamp'], unit='ms')

# Normalize to same date
fear_greed['date'] = fear_greed['date'].dt.floor('D')
trades['date'] = trades['date'].dt.floor('D')

# -----------------------
# 4. CLEAN NUMERIC DATA
# -----------------------

numeric_cols = ['execution_price', 'size_tokens', 'size_usd', 'closed_pnl', 'Fee']

for col in numeric_cols:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

# Drop invalid rows
trades = trades.dropna(subset=['closed_pnl'])

# -----------------------
# 5. FEATURE ENGINEERING
# -----------------------

trades['is_win'] = trades['closed_pnl'] > 0

# Use USD size (better than tokens)
trades['trade_volume'] = trades['size_usd']

# -----------------------
# 6. AGGREGATE DAILY TRADER STATS
# -----------------------

daily_stats = trades.groupby(['account', 'date']).agg(
    total_pnl=('closed_pnl', 'sum'),
    avg_pnl=('closed_pnl', 'mean'),
    win_rate=('is_win', 'mean'),
    trade_count=('closed_pnl', 'count'),
    total_volume=('trade_volume', 'sum')
).reset_index()

# -----------------------
# 7. SENTIMENT MAPPING
# -----------------------

sentiment_map = {
    'Fear': -1,
    'Extreme Fear': -2,
    'Neutral': 0,
    'Greed': 1,
    'Extreme Greed': 2
}

fear_greed['sentiment_score'] = fear_greed['Classification'].map(sentiment_map)

# -----------------------
# 8. MERGE DATASETS
# -----------------------

merged = pd.merge(
    daily_stats,
    fear_greed[['date', 'Classification', 'sentiment_score']],
    on='date',
    how='inner'
)

print("Merged rows:", len(merged))

print(trades[['timestamp', 'date']].head())
print(fear_greed[['timestamp', 'date']].head())
# -----------------------
# 9. ANALYSIS
# -----------------------

sentiment_perf = merged.groupby('Classification').agg(
    avg_total_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trades=('trade_count', 'mean'),
    samples=('account', 'count')
).sort_values(by='avg_total_pnl', ascending=False)

print("\n=== Performance by Sentiment ===")
print(sentiment_perf)

# -----------------------
# 10. CORRELATION
# -----------------------

corr = merged[['total_pnl', 'win_rate', 'trade_count', 'sentiment_score']].corr()
print("\n=== Correlation ===")
print(corr)

# -----------------------
# 11. VISUALIZATION
# -----------------------

# PnL vs Sentiment
plt.figure()
merged.boxplot(column='total_pnl', by='Classification', rot=45)
plt.title("PnL by Market Sentiment")
plt.suptitle("")
plt.tight_layout()
plt.show()

# Win rate vs Sentiment
plt.figure()
merged.boxplot(column='win_rate', by='Classification', rot=45)
plt.title("Win Rate by Market Sentiment")
plt.suptitle("")
plt.tight_layout()
plt.show()

# -----------------------
# 12. OPTIONAL MODEL
# -----------------------

from sklearn.linear_model import LinearRegression

model_data = merged.dropna(subset=['sentiment_score'])

X = model_data[['sentiment_score']]
y = model_data['total_pnl']

model = LinearRegression()
model.fit(X, y)

print("\n=== Model Insight ===")
print(f"Sentiment Impact Coef: {model.coef_[0]:.4f}")

# -----------------------
# 13. SAVE OUTPUT
# -----------------------

merged.to_csv("merged_output.csv", index=False)
sentiment_perf.to_csv("sentiment_summary.csv")