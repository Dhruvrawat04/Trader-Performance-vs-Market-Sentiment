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

### Step 1

Run preprocessing

```bash
python main.py
```

This generates the cleaned and aligned daily datasets.

---

### Step 2

Run analysis

```bash
python analysis.py
```

This generates

- summary tables
- charts
- trader segmentation

Outputs are saved in

```
analysis_outputs/
```

---

### Step 3 (Optional)

Run strategy recommendations and predictive model

```bash
python strategy.py
```

This prints

- strategy recommendations
- Random Forest prediction accuracy

---

## Outputs

Generated files include

- sentiment_comparison.png
- segment_comparison.png
- sentiment_summary.csv
- behavior_summary.csv
- segment tables

---

## Libraries Used

- Pandas
- Matplotlib
- Scikit-Learn