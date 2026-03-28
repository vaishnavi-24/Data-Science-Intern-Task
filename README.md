 ##Crypto Trader Sentiment Analysis

##Overview

This project analyzes the relationship between **Bitcoin market sentiment (Fear & Greed Index)** and **trader performance** using Hyperliquid historical trading data.

Goal: uncover whether sentiment impacts profitability and identify patterns for smarter trading strategies.

---

## Datasets

### 1. Fear & Greed Index

* Columns: `timestamp`, `value`, `classification`, `date`
* Timestamp format: **UNIX seconds**

### 2. Historical Trader Data

* Columns include:

  * `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`
  * `Side`, `Timestamp`, `Closed PnL`, `Fee`, etc.
* Timestamp format: **UNIX milliseconds**

---

##  Key Challenges

* Mismatched timestamp formats (seconds vs milliseconds)
* Missing / inconsistent data
* Aligning time-series datasets

---

## Methodology

### 1. Data Preprocessing

* Convert timestamps → datetime
* Normalize dates (daily granularity)
* Clean numeric fields

### 2. Feature Engineering

* `total_pnl`, `avg_pnl`
* `win_rate`
* `trade_count`
* `total_volume`

### 3. Data Merge

* Join trader data with sentiment by date

### 4. Analysis

* Performance grouped by sentiment
* Correlation analysis

### 5. Advanced Analysis

* Sharpe Ratio (risk-adjusted return)
* Max Drawdown (risk exposure)
* Trader segmentation:

  * Top 10%
  * Bottom 10%
  * Whales

### 6. Predictive Modeling

* Lag feature: sentiment → next-day PnL
* Model: Random Forest Classifier
* Target: next-day profitability

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Run base analysis

```bash
python main1.py
```

### 3. Run advanced analysis

```bash
python main.py
```

---

## Outputs

* Performance by sentiment
* Correlation matrix
* Risk metrics per trader
* ML model results

---

##  Future Improvements

* XGBoost / LightGBM models
* Time-series modeling (LSTM)
* Dashboard (Streamlit)
* Strategy backtesting

---

##  Takeaway

Handling real-world financial data requires:

* Strong data cleaning
* Correct time alignment
* Thoughtful feature engineering

This project demonstrates practical **quant + data science skills** applied to crypto markets.

---

## 📬 Contact

Feel free to connect or reach out for collaboration!
