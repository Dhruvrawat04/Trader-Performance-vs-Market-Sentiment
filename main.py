from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


def dataset_quality_report(name: str, frame: pd.DataFrame) -> None:
	print(f"{name} Shape: {frame.shape}")
	print(f"{name} Missing Values")
	print(frame.isna().sum().to_string())
	print(f"{name} Duplicates: {frame.duplicated().sum()}")


def main() -> None:
	fear_greed = pd.read_csv(DATASET_DIR / "fear_greed_index.csv")
	historical = pd.read_csv(DATASET_DIR / "historical_data.csv")

	dataset_quality_report("Fear & Greed", fear_greed)
	print()
	dataset_quality_report("Historical", historical)

	fear_greed["date"] = pd.to_datetime(fear_greed["date"], errors="coerce").dt.normalize()
	historical["trade_datetime"] = pd.to_datetime(
		historical["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
	)
	historical["trade_date"] = historical["trade_datetime"].dt.normalize()

	historical["size_usd"] = pd.to_numeric(historical["Size USD"], errors="coerce")
	historical["closed_pnl"] = pd.to_numeric(historical["Closed PnL"], errors="coerce")
	historical["start_position"] = pd.to_numeric(historical["Start Position"], errors="coerce")
	historical["fee"] = pd.to_numeric(historical["Fee"], errors="coerce")

	historical["trade_result"] = historical["closed_pnl"] > 0
	historical["long_trade"] = historical["Direction"].astype(str).str.contains("Long", case=False, na=False)
	historical["short_trade"] = historical["Direction"].astype(str).str.contains("Short", case=False, na=False)
	historical["leverage_proxy"] = historical["start_position"].abs().div(historical["size_usd"].replace(0, pd.NA))

	daily_metrics = (
		historical.groupby(["Account", "trade_date"], dropna=False)
		.agg(
			daily_pnl=("closed_pnl", "sum"),
			win_rate=("trade_result", "mean"),
			avg_trade_size_usd=("size_usd", "mean"),
			num_trades=("Trade ID", "count"),
			long_trades=("long_trade", "sum"),
			short_trades=("short_trade", "sum"),
			avg_leverage_proxy=("leverage_proxy", "mean"),
			total_fees=("fee", "sum"),
		)
		.reset_index()
	)
	daily_metrics["long_short_ratio"] = daily_metrics["long_trades"].div(
		daily_metrics["short_trades"].replace(0, pd.NA)
	)

	daily_market = fear_greed[["date", "value", "classification"]].drop_duplicates(subset=["date"])
	aligned_daily = daily_metrics.merge(
		daily_market,
		left_on="trade_date",
		right_on="date",
		how="left",
	)

	overall_daily = (
		historical.groupby("trade_date", dropna=False)
		.agg(
			daily_pnl=("closed_pnl", "sum"),
			win_rate=("trade_result", "mean"),
			avg_trade_size_usd=("size_usd", "mean"),
			num_trades=("Trade ID", "count"),
			long_trades=("long_trade", "sum"),
			short_trades=("short_trade", "sum"),
			avg_leverage_proxy=("leverage_proxy", "mean"),
			total_fees=("fee", "sum"),
		)
		.reset_index()
	)
	overall_daily["long_short_ratio"] = overall_daily["long_trades"].div(
		overall_daily["short_trades"].replace(0, pd.NA)
	)

	overall_aligned = overall_daily.merge(daily_market, left_on="trade_date", right_on="date", how="left")

	print("\nAligned daily metrics preview")
	print(aligned_daily.head().to_string(index=False))

	print("\nOverall daily metrics preview")
	print(overall_aligned.head().to_string(index=False))

	print("\nCoverage checks")
	print("Historical rows with parsed date:", historical["trade_date"].notna().sum())
	print("Fear & Greed rows with parsed date:", fear_greed["date"].notna().sum())
	print("Aligned daily rows:", len(aligned_daily))


if __name__ == "__main__":
	main()
