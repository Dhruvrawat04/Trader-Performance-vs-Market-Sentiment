import pandas as pd

from strategy import get_trade_signal, load_daily_data, run_backtest


def test_get_trade_signal_classifies_sentiment() -> None:
    assert get_trade_signal("Greed") == 1.0
    assert get_trade_signal("Fear") == 0.0
    assert get_trade_signal("Neutral") == 0.5


def test_run_backtest_returns_expected_metrics() -> None:
    daily = pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "daily_pnl": [10.0, -5.0, 4.0],
            "sentiment": ["Greed", "Fear", "Greed"],
        }
    )

    result = run_backtest(daily)

    assert result["final_equity"].iloc[0] == 19.0
    assert result["total_pnl"].iloc[0] == 19.0
    assert result["win_rate"].iloc[0] == 1.0


def test_load_daily_data_creates_sentiment_from_classification() -> None:
    daily = load_daily_data()

    assert "sentiment" in daily.columns
    assert daily["sentiment"].notna().all()
