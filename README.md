# Bitcoin Market Sentiment vs Trader Performance Analysis

## Overview

This project investigates whether Bitcoin market sentiment (Fear & Greed Index) affects trader performance and trading behaviour.

The analysis combines historical trading data with the Fear & Greed Index and evaluates differences in profitability, trading behaviour, trader segmentation, and simple predictive modelling.

---

## Dataset

- fear_greed_index.csv
- historical_data.csv

---

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Run the full pipeline

```bash
python main.py
python analysis.py
python strategy.py
```

This generates the cleaned datasets, analysis outputs, charts, and a simple sentiment-based backtest.

### Option 2: Run everything with one command

```bash
python -m pytest
```

The test suite checks the new strategy functions and the backtest logic.

---

## Outputs

Generated files include

- sentiment_comparison.png
- segment_comparison.png
- sentiment_summary.csv
- behavior_summary.csv
- segment tables
- strategy_backtest.csv

---

## Outputs

Generated files include

- sentiment_comparison.png
- segment_comparison.png
- sentiment_summary.csv
- behavior_summary.csv
- segment tables
- strategy_backtest.csv

---

## Libraries Used

- Pandas
- Matplotlib
- Scikit-Learn

