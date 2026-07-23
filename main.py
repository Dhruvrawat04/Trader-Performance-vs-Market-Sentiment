from pathlib import Path

import pandas as pd

from data_utils import load_raw_data, prepare_data

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


def dataset_quality_report(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name} Dataset Report")
    print("=" * 60)
    print(f"Shape: {frame.shape[0]} rows × {frame.shape[1]} columns")
    print("\nMissing Values:")
    print(frame.isna().sum().to_string())
    print(f"\nDuplicate Rows: {frame.duplicated().sum()}")
    print("\nFirst 5 Rows:")
    print(frame.head().to_string(index=False))


def main() -> None:
    fear_greed, historical = load_raw_data()
    dataset_quality_report("Fear & Greed", fear_greed)
    dataset_quality_report("Historical", historical)

    _, _, account_daily, overall_daily = prepare_data()

    account_daily["win_rate"] *= 100
    overall_daily["win_rate"] *= 100

    print("\n" + "=" * 60)
    print("Daily Metrics Per Account")
    print("=" * 60)
    print(account_daily.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Overall Daily Metrics")
    print("=" * 60)
    print(overall_daily.head(10).to_string(index=False))

    account_daily.to_csv(DATASET_DIR / "aligned_daily_metrics.csv", index=False)
    overall_daily.to_csv(DATASET_DIR / "overall_daily_metrics.csv", index=False)

    print("\nProcessed datasets saved successfully.")


if __name__ == "__main__":
    main()
