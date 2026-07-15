from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fear_greed = pd.read_csv(DATASET_DIR / "fear_greed_index.csv")
    historical = pd.read_csv(DATASET_DIR / "historical_data.csv")
    return fear_greed, historical


def prepare_daily_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    fear_greed, historical = load_data()

    fear_greed["date"] = pd.to_datetime(fear_greed["date"], errors="coerce").dt.normalize()
    fear_greed["sentiment"] = fear_greed["classification"].replace(
        {
            "Extreme Fear": "Fear",
            "Fear": "Fear",
            "Neutral": "Neutral",
            "Greed": "Greed",
            "Extreme Greed": "Greed",
        }
    )

    historical["trade_datetime"] = pd.to_datetime(
        historical["Timestamp IST"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    historical["trade_date"] = historical["trade_datetime"].dt.normalize()

    historical["size_usd"] = pd.to_numeric(historical["Size USD"], errors="coerce")
    historical["closed_pnl"] = pd.to_numeric(historical["Closed PnL"], errors="coerce")
    historical["start_position"] = pd.to_numeric(historical["Start Position"], errors="coerce")
    historical["fee"] = pd.to_numeric(historical["Fee"], errors="coerce")

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
        )
        .reset_index()
    )
    overall_daily["long_short_ratio"] = overall_daily["long_trades"].div(
        overall_daily["short_trades"].replace(0, pd.NA)
    )
    overall_daily = overall_daily.merge(daily_market, left_on="trade_date", right_on="date", how="left")

    return account_daily, overall_daily


def print_rules_of_thumb(account_daily: pd.DataFrame, overall_daily: pd.DataFrame) -> None:
    account_segment = account_daily[
        account_daily["sentiment"].isin(["Fear", "Greed"])
    ].copy()

    leverage_threshold = account_segment["avg_position_size_ratio"].median()
    activity_threshold = account_segment["num_trades"].median()

    account_segment["leverage_segment"] = account_segment[
        "avg_position_size_ratio"
    ].apply(lambda x: "High" if x >= leverage_threshold else "Low")

    account_segment["activity_segment"] = account_segment[
        "num_trades"
    ].apply(lambda x: "Frequent" if x >= activity_threshold else "Infrequent")

    fear_high = account_segment[
        (account_segment["sentiment"] == "Fear")
        & (account_segment["leverage_segment"] == "High")
    ]["daily_pnl"].mean()

    fear_low = account_segment[
        (account_segment["sentiment"] == "Fear")
        & (account_segment["leverage_segment"] == "Low")
    ]["daily_pnl"].mean()

    greed_frequent = account_segment[
        (account_segment["sentiment"] == "Greed")
        & (account_segment["activity_segment"] == "Frequent")
    ]["win_rate"].mean()

    greed_infrequent = account_segment[
        (account_segment["sentiment"] == "Greed")
        & (account_segment["activity_segment"] == "Infrequent")
    ]["win_rate"].mean()

    print("\nStrategy Ideas")

    if fear_high < fear_low:
        print(
            f"1. During Fear days, reduce leverage because "
            f"high-leverage traders earned lower average daily PnL "
            f"({fear_high:.2f}) than low-leverage traders ({fear_low:.2f})."
        )
    else:
        print(
            f"1. During Fear days, high-leverage traders achieved "
            f"higher average daily PnL ({fear_high:.2f}) than low-leverage "
            f"traders ({fear_low:.2f}); reducing leverage is not supported by this dataset."
        )

    if greed_frequent > greed_infrequent:
        print(
            f"2. During Greed days, allow higher trading activity for "
            f"frequent traders because their win rate "
            f"({greed_frequent:.2%}) exceeded infrequent traders "
            f"({greed_infrequent:.2%})."
        )
    else:
        print(
            f"2. During Greed days, increasing trade frequency is not supported "
            f"because frequent traders had a lower win rate "
            f"({greed_frequent:.2%}) than infrequent traders "
            f"({greed_infrequent:.2%})."
        )


def run_simple_predictive_model(overall_daily: pd.DataFrame) -> None:

    daily = overall_daily.sort_values("trade_date").copy()

    daily["prev_day_pnl"] = daily["daily_pnl"].shift(1)
    daily["prev_day_win_rate"] = daily["win_rate"].shift(1)

    daily = daily.dropna(
        subset=[
            "value",
            "num_trades",
            "avg_trade_size_usd",
            "avg_position_size_ratio",
            "long_short_ratio",
            "prev_day_pnl",
            "prev_day_win_rate",
        ]
    )

    daily["target"] = (daily["daily_pnl"] > 0).astype(int)

    if len(daily) < 20 or daily["target"].nunique() < 2:
        print("\nNot enough data for prediction.")
        return

    X = daily[
        [
            "value",
            "num_trades",
            "avg_trade_size_usd",
            "avg_position_size_ratio",
            "long_short_ratio",
            "prev_day_pnl",
            "prev_day_win_rate",
        ]
    ]

    y = daily["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print("\nBonus Predictive Model")
    print(
        f"Accuracy: {accuracy_score(y_test, prediction):.3f}"
    )

    importance = (
        pd.Series(
            model.feature_importances_,
            index=X.columns,
        )
        .sort_values(ascending=False)
    )

    print("\nFeature Importance")
    print(importance.round(3))


def main() -> None:
    account_daily, overall_daily = prepare_daily_features()
    print_rules_of_thumb(account_daily, overall_daily)
    run_simple_predictive_model(overall_daily)


if __name__ == "__main__":
    main()