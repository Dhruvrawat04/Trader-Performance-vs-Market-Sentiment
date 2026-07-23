# 📈 Bitcoin Market Sentiment vs Trader Performance Analysis

## 🎯 Overview

This project studies whether Bitcoin market sentiment, measured by the Fear & Greed Index, influences trader performance and trading behavior.

It combines historical trading data with market sentiment data to explore:

- profitability differences across sentiment regimes
- changes in trader behavior
- trader segmentation based on risk and activity
- a simple sentiment-based backtest

---

## 📂 Dataset

The project uses these input files:

- [dataset/fear_greed_index.csv](dataset/fear_greed_index.csv)
- [dataset/historical_data.csv](dataset/historical_data.csv)

---

## ⚙️ Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1: Run the full pipeline

```bash
python pipeline.py
```

This runs the preprocessing, analysis, charts, and backtest in one step.

### Option 2: Run the scripts individually

```bash
python main.py
python analysis.py
python strategy.py
```

### Option 3: Run tests

```bash
python -m pytest
```

---

## 📦 Outputs

The pipeline generates files such as:

- sentiment_comparison.png
- segment_comparison.png
- sentiment_summary.csv
- behavior_summary.csv
- segment_high_vs_low_leverage.csv
- segment_frequent_vs_infrequent.csv
- segment_consistent_vs_inconsistent.csv
- strategy_backtest.csv

These outputs are stored in [analysis_outputs](analysis_outputs) and the processed datasets in [dataset](dataset).

---

## 🧰 Libraries Used

- Pandas
- Matplotlib
- Scikit-Learn
- pytest

