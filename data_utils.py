from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


def normalize_sentiment(value: object) -> str:
    if pd.isna(value):
        return "Neutral"
    mapping = {
        "Extreme Fear": "Fear",
        "Fear": "Fear",
        "Neutral": "Neutral",
        "Greed": "Greed",
        "Extreme Greed": "Greed",
    }
    return mapping.get(str(value), "Neutral")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fear_greed = pd.read_csv(DATASET_DIR / "fear_greed_index.csv")
    historical = pd.read_csv(DATASET_DIR / "historical_data.csv")
    return fear_greed, historical


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fear_greed, historical = load_raw_data()

    fear_greed["date"] = pd.to_datetime(fear_greed["date"], errors="coerce").dt.normalize()
    fear_greed["sentiment"] = fear_greed["classification"].apply(normalize_sentiment)

    historical["trade_datetime"] = pd.to_datetime(
        historical["Timestamp IST"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    historical["trade_date"] = historical["trade_datetime"].dt.normalize()

    for original, renamed in {
        "Size USD": "size_usd",
        "Closed PnL": "closed_pnl",
        "Start Position": "start_position",
        "Fee": "fee",
    }.items():
        historical[renamed] = pd.to_numeric(historical[original], errors="coerce")

    historical["trade_result"] = historical["closed_pnl"] > 0
    historical["long_trade"] = historical["Direction"].astype(str).str.contains("Long", case=False, na=False)
    historical["short_trade"] = historical["Direction"].astype(str).str.contains("Short", case=False, na=False)
    historical["position_size_ratio"] = historical["start_position"].abs().div(
        historical["size_usd"].replace(0, pd.NA)
    )

    daily_market = fear_greed[["date", "value", "classification", "sentiment"]].drop_duplicates(subset=["date"])

    account_daily = (
        historical.groupby(["Account", "trade_date"], dropna=False)
        .agg(
            daily_pnl=("closed_pnl", "sum"),
            win_rate=("trade_result", "mean"),
            avg_trade_size_usd=("size_usd", "mean"),
            num_trades=("Trade ID", "count"),
            long_trades=("long_trade", "sum"),
            short_trades=("short_trade", "sum"),
            avg_position_size_ratio=("position_size_ratio", "mean"),
            total_fees=("fee", "sum"),
        )
        .reset_index()
    )
    account_daily["long_short_ratio"] = account_daily["long_trades"].div(
        account_daily["short_trades"].replace(0, pd.NA)
    )
    account_daily = account_daily.merge(daily_market, left_on="trade_date", right_on="date", how="left")

    overall_daily = (
        historical.groupby("trade_date", dropna=False)
        .agg(
            daily_pnl=("closed_pnl", "sum"),
            win_rate=("trade_result", "mean"),
            avg_trade_size_usd=("size_usd", "mean"),
            num_trades=("Trade ID", "count"),
            long_trades=("long_trade", "sum"),
            short_trades=("short_trade", "sum"),
            avg_position_size_ratio=("position_size_ratio", "mean"),
            total_fees=("fee", "sum"),
        )
        .reset_index()
    )
    overall_daily["long_short_ratio"] = overall_daily["long_trades"].div(
        overall_daily["short_trades"].replace(0, pd.NA)
    )
    overall_daily = overall_daily.merge(daily_market, left_on="trade_date", right_on="date", how="left")

    return fear_greed, historical, account_daily, overall_daily
