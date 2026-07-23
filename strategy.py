from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_utils import normalize_sentiment

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"


def load_daily_data() -> pd.DataFrame:
    overall_daily = pd.read_csv(DATASET_DIR / "overall_daily_metrics.csv")
    overall_daily["trade_date"] = pd.to_datetime(overall_daily["trade_date"], errors="coerce")

    if "sentiment" not in overall_daily.columns:
        if "classification" in overall_daily.columns:
            overall_daily["sentiment"] = overall_daily["classification"].apply(normalize_sentiment)
        else:
            overall_daily["sentiment"] = "Neutral"

    overall_daily = overall_daily.sort_values("trade_date").reset_index(drop=True)
    return overall_daily


def get_trade_signal(sentiment: str) -> float:
    if sentiment == "Greed":
        return 1.0
    if sentiment == "Fear":
        return 0.0
    return 0.5


def run_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["signal"] = daily["sentiment"].fillna("Neutral").apply(get_trade_signal)
    daily["position"] = daily["signal"].replace({0.0: -1.0, 0.5: 0.0, 1.0: 1.0})
    daily["strategy_pnl"] = daily["position"] * daily["daily_pnl"]
    daily["equity_curve"] = daily["strategy_pnl"].cumsum()

    summary = pd.DataFrame(
        {
            "final_equity": [daily["equity_curve"].iloc[-1]],
            "total_pnl": [daily["strategy_pnl"].sum()],
            "win_rate": [daily.loc[daily["strategy_pnl"] > 0, "strategy_pnl"].shape[0] / max(1, daily["strategy_pnl"].shape[0])],
        }
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    daily = load_daily_data()
    if daily.empty:
        raise ValueError("No daily data available for backtest")

    summary = run_backtest(daily)
    summary.to_csv(OUTPUT_DIR / "strategy_backtest.csv", index=False)

    print("Strategy Backtest")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
