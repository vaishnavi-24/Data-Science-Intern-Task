# file: analysis/advanced_trader_analysis.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------
# 1. LOAD PRE-MERGED DATA
# -----------------------
merged = pd.read_csv("merged_output.csv")

# Ensure datetime
merged['date'] = pd.to_datetime(merged['date'])

# -----------------------
# 2. RISK METRICS
# -----------------------

def compute_sharpe(returns):
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def compute_drawdown(pnl_series):
    cumulative = pnl_series.cumsum()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak)
    return drawdown.min()

# Compute per trader
risk_metrics = merged.groupby('account').apply(
    lambda df: pd.Series({
        'sharpe': compute_sharpe(df['total_pnl']),
        'max_drawdown': compute_drawdown(df['total_pnl']),
        'total_pnl': df['total_pnl'].sum(),
        'avg_volume': df['total_volume'].mean()
    })
).reset_index()

# -----------------------
# 3. TRADER SEGMENTATION
# -----------------------

# Top / Bottom traders
top_threshold = risk_metrics['total_pnl'].quantile(0.9)
bottom_threshold = risk_metrics['total_pnl'].quantile(0.1)

risk_metrics['segment'] = 'mid'
risk_metrics.loc[risk_metrics['total_pnl'] >= top_threshold, 'segment'] = 'top_10%'
risk_metrics.loc[risk_metrics['total_pnl'] <= bottom_threshold, 'segment'] = 'bottom_10%'

# Whale detection (top volume)
volume_threshold = risk_metrics['avg_volume'].quantile(0.9)
risk_metrics['is_whale'] = risk_metrics['avg_volume'] >= volume_threshold

print("\n=== Trader Segments ===")
print(risk_metrics['segment'].value_counts())

# -----------------------
# 4. LAGGED SENTIMENT
# -----------------------

merged = merged.sort_values(['account', 'date'])

merged['next_day_pnl'] = merged.groupby('account')['total_pnl'].shift(-1)
merged['next_day_profitable'] = merged['next_day_pnl'] > 0

# -----------------------
# 5. FEATURE SET
# -----------------------

model_data = merged.dropna(subset=[
    'sentiment_score',
    'next_day_profitable'
])

X = model_data[['sentiment_score', 'win_rate', 'trade_count', 'total_volume']]
y = model_data['next_day_profitable']

# -----------------------
# 6. TRAIN MODEL
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# -----------------------
# 7. FEATURE IMPORTANCE
# -----------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# -----------------------
# 8. INSIGHTS EXPORT
# -----------------------

risk_metrics.to_csv("trader_risk_metrics.csv", index=False)
model_data.to_csv("model_dataset.csv", index=False)