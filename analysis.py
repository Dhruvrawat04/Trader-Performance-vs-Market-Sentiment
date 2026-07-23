from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import prepare_data


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis_outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def add_drawdown_proxy(daily_frame: pd.DataFrame) -> pd.DataFrame:
    frame = daily_frame.sort_values("trade_date").copy()
    frame["equity_curve"] = frame["daily_pnl"].cumsum()
    frame["running_peak"] = frame["equity_curve"].cummax()
    frame["drawdown_proxy"] = frame["equity_curve"] - frame["running_peak"]
    return frame


def compare_sentiment_performance(overall_daily: pd.DataFrame) -> pd.DataFrame:
    daily = overall_daily[overall_daily["sentiment"].isin(["Fear", "Greed"])].copy()
    daily = add_drawdown_proxy(daily)

    sentiment_summary = (
        daily.groupby("sentiment")
        .agg(
            days=("trade_date", "nunique"),
            avg_daily_pnl=("daily_pnl", "mean"),
            median_daily_pnl=("daily_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_drawdown_proxy=("drawdown_proxy", "mean"),
            worst_drawdown_proxy=("drawdown_proxy", "min"),
            avg_trade_count=("num_trades", "mean"),
        )
        .reset_index()
    )
    return daily, sentiment_summary


def trader_behavior_by_sentiment(account_daily: pd.DataFrame) -> pd.DataFrame:
    daily = account_daily.copy()
    daily = daily[daily["sentiment"].isin(["Fear", "Greed"])]
    daily["long_share"] = daily["long_trades"].div((daily["long_trades"] + daily["short_trades"]).replace(0, pd.NA))

    behavior = (
        daily.groupby("sentiment")
        .agg(
            avg_trade_frequency=("num_trades", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_position_size_ratio=("avg_position_size_ratio", "mean"),
            avg_long_short_ratio=("long_short_ratio", "mean"),
            avg_long_share=("long_share", "mean"),
            avg_daily_pnl=("daily_pnl", "mean"),
            avg_win_rate=("win_rate", "mean"),
        )
        .reset_index()
    )
    return behavior


def make_segment_tables(account_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    account_summary = (
        account_daily.groupby("Account")
        .agg(
            avg_daily_pnl=("daily_pnl", "mean"),
            pnl_std=("daily_pnl", "std"),
            win_rate=("win_rate", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_position_size_ratio=("avg_position_size_ratio", "mean"),
            avg_num_trades=("num_trades", "mean"),
            avg_long_share=("long_trades", lambda s: s.sum()),
            avg_short_share=("short_trades", lambda s: s.sum()),
        )
        .reset_index()
    )

    account_summary["trade_total"] = account_summary["avg_long_share"] + account_summary["avg_short_share"]
    account_summary["long_share"] = account_summary["avg_long_share"].div(account_summary["trade_total"].replace(0, pd.NA))
    account_summary["pnl_consistency"] = account_summary["avg_daily_pnl"].div(account_summary["pnl_std"].replace(0, pd.NA))

    leverage_threshold = account_summary["avg_position_size_ratio"].median()
    frequency_threshold = account_summary["avg_num_trades"].median()
    win_threshold = account_summary["win_rate"].median()

    segments = {
        "high_vs_low_leverage": account_summary.assign(
            segment=lambda df: pd.Series(
                ["High leverage" if x >= leverage_threshold else "Low leverage" for x in df["avg_position_size_ratio"]],
                index=df.index,
            )
        ),
        "frequent_vs_infrequent": account_summary.assign(
            segment=lambda df: pd.Series(
                ["Frequent" if x >= frequency_threshold else "Infrequent" for x in df["avg_num_trades"]],
                index=df.index,
            )
        ),
        "consistent_vs_inconsistent": account_summary.assign(
            segment=lambda df: pd.Series(
                ["Consistent winner" if (w >= win_threshold and p >= 0) else "Inconsistent" for w, p in zip(df["win_rate"], df["avg_daily_pnl"])],
                index=df.index,
            )
        ),
    }

    for key, summary in list(segments.items()):
        segment_table = (
            summary.groupby("segment")
            .agg(
                traders=("Account", "nunique"),
                avg_daily_pnl=("avg_daily_pnl", "mean"),
                avg_win_rate=("win_rate", "mean"),
                avg_trade_size_usd=("avg_trade_size_usd", "mean"),
                avg_position_size_ratio=("avg_position_size_ratio", "mean"),
                avg_trade_frequency=("avg_num_trades", "mean"),
                avg_long_share=("long_share", "mean"),
            )
            .reset_index()
        )
        segments[key] = segment_table

    return segments


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def plot_sentiment_comparison(sentiment_summary: pd.DataFrame, behavior_summary: pd.DataFrame, daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fear vs Greed Performance and Behavior", fontsize=16, fontweight="bold")

    axes[0, 0].bar(sentiment_summary["sentiment"], sentiment_summary["avg_daily_pnl"], color=["#4C78A8", "#F58518"])
    axes[0, 0].set_title("Average Daily PnL")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("PnL")

    axes[0, 1].bar(sentiment_summary["sentiment"], sentiment_summary["avg_win_rate"], color=["#54A24B", "#E45756"])
    axes[0, 1].set_title("Average Win Rate")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Win rate")

    axes[1, 0].bar(behavior_summary["sentiment"], behavior_summary["avg_trade_frequency"], color=["#72B7B2", "#9D755D"])
    axes[1, 0].set_title("Average Trades per Account-Day")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Trades")

    sentiment_colors = {"Fear": "#4C78A8", "Greed": "#F58518"}
    for sentiment, subset in daily.groupby("sentiment"):
        subset = subset.dropna(subset=["long_short_ratio", "daily_pnl"])
        axes[1, 1].scatter(
            subset["long_short_ratio"],
            subset["daily_pnl"],
            alpha=0.6,
            s=18,
            label=sentiment,
            color=sentiment_colors.get(sentiment, "#666666"),
        )
    axes[1, 1].set_title("Long/Short Bias vs Daily PnL")
    axes[1, 1].set_xlabel("Long/Short Ratio")
    axes[1, 1].set_ylabel("Daily PnL")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_segments(segment_tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Trader Segments", fontsize=16, fontweight="bold")

    for ax, (title, table) in zip(axes, segment_tables.items()):
        ax.bar(table["segment"], table["avg_daily_pnl"], color=["#4C78A8", "#F58518"][: len(table)])
        ax.set_title(title.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Average Daily PnL")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    ensure_output_dir()

    fear_greed, historical, account_daily, overall_daily = prepare_data()

    daily, sentiment_summary = compare_sentiment_performance(overall_daily)
    behavior_summary = trader_behavior_by_sentiment(account_daily)
    segment_tables = make_segment_tables(account_daily)

    save_tables(
        {
            "sentiment_summary": sentiment_summary,
            "behavior_summary": behavior_summary,
            "segment_high_vs_low_leverage": segment_tables["high_vs_low_leverage"],
            "segment_frequent_vs_infrequent": segment_tables["frequent_vs_infrequent"],
            "segment_consistent_vs_inconsistent": segment_tables["consistent_vs_inconsistent"],
        }
    )

    plot_sentiment_comparison(sentiment_summary, behavior_summary, daily)
    plot_segments(segment_tables)
    print("\nFear vs Greed Performance")
    print(sentiment_summary)

    print("\nTrader Behaviour")
    print(behavior_summary)

    print("\nHigh vs Low Leverage")
    print(segment_tables["high_vs_low_leverage"])

    print("\nFrequent vs Infrequent")
    print(segment_tables["frequent_vs_infrequent"])

    print("\nConsistent vs Inconsistent")
    print(segment_tables["consistent_vs_inconsistent"])
    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()