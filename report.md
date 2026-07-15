# Methodology

Two datasets were used:

1. Bitcoin Fear & Greed Index
2. Historical trader activity

The timestamps of both datasets were converted into datetime format and aligned at the daily level.

Several trading metrics were created, including:

- Daily PnL
- Win Rate
- Average Trade Size
- Position Size Ratio
- Number of Trades
- Long/Short Ratio

Trader behaviour was compared across Fear and Greed market conditions.

Three trader segments were also analysed:

- High vs Low Position Size Ratio
- Frequent vs Infrequent Traders
- Consistent vs Inconsistent Traders

Finally, a Random Forest classifier was trained to predict whether the next trading day would be profitable using sentiment and trading behaviour features.

---

# Key Insights

1. Trader profitability differed between Fear and Greed market conditions, indicating that market sentiment influences trading outcomes.

2. Trading behaviour changed with sentiment. Trade frequency, position sizes and long/short positioning varied between Fear and Greed periods.

3. High-risk traders behaved differently from conservative traders. Segment analysis showed measurable differences in profitability, win rate and trading frequency.

---

# Strategy Recommendations

### Strategy 1

During periods of Fear, reduce exposure for traders operating with high position-size ratios when historical performance shows lower profitability.

### Strategy 2

Increase trading activity only for trader segments that historically maintain a higher win rate during Greed periods.

These recommendations are based on the observed historical relationships between market sentiment and trader performance.

---

# Bonus Model

A Random Forest classifier was trained using:

- Fear & Greed Index
- Trade Frequency
- Average Trade Size
- Position Size Ratio
- Long/Short Ratio
- Previous Day PnL
- Previous Day Win Rate

The model predicts whether the following trading day will be profitable and reports classification accuracy together with feature importance.