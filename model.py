from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_utils import prepare_data


BASE_DIR = Path(__file__).resolve().parent


def prepare_daily_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, account_daily, overall_daily = prepare_data()
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