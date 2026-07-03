from pathlib import Path

import pandas as pd


TIMELINE_PATH = Path("data/interim/timelines/cgm_timeline.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "cgm_lag_dataset.csv"

LAG_MINUTES = list(range(0, 125, 5))  # 0, 5, 10, ..., 120
FORECAST_HORIZONS = [30, 60]


def load_timeline() -> pd.DataFrame:
    if not TIMELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TIMELINE_PATH}. Run src/data/build_cgm_timeline.py first."
        )

    df = pd.read_csv(TIMELINE_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    required_cols = {"patient_id", "split", "timestamp", "glucose"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns from CGM timeline: {missing}")

    df = df.dropna(subset=["timestamp", "glucose"])
    df = df.sort_values(["patient_id", "split", "timestamp"]).reset_index(drop=True)

    return df


def add_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()

    # Since timeline is already 5-min resampled, one shift = 5 minutes.
    for minutes in LAG_MINUTES:
        steps = minutes // 5
        group[f"glucose_lag_{minutes}min"] = group["glucose"].shift(steps)

    for horizon in FORECAST_HORIZONS:
        steps = horizon // 5
        group[f"target_glucose_{horizon}min"] = group["glucose"].shift(-steps)

    # Simple trend features from CGM only.
    group["glucose_change_30min"] = (
        group["glucose_lag_0min"] - group["glucose_lag_30min"]
    )
    group["glucose_change_60min"] = (
        group["glucose_lag_0min"] - group["glucose_lag_60min"]
    )
    group["glucose_change_120min"] = (
        group["glucose_lag_0min"] - group["glucose_lag_120min"]
    )

    group["glucose_rolling_mean_30min"] = (
        group["glucose"].rolling(window=6, min_periods=6).mean()
    )
    group["glucose_rolling_std_30min"] = (
        group["glucose"].rolling(window=6, min_periods=6).std()
    )
    group["glucose_rolling_mean_60min"] = (
        group["glucose"].rolling(window=12, min_periods=12).mean()
    )
    group["glucose_rolling_std_60min"] = (
        group["glucose"].rolling(window=12, min_periods=12).std()
    )

    return group


def main() -> None:
    timeline = load_timeline()

    datasets = []

    for (patient_id, split), group in timeline.groupby(["patient_id", "split"], sort=True):
        print(f"Building lag features for patient={patient_id}, split={split}")
        datasets.append(add_lag_features(group))

    dataset = pd.concat(datasets, ignore_index=True)

    feature_cols = [
        col
        for col in dataset.columns
        if col.startswith("glucose_lag_")
        or col.startswith("glucose_change_")
        or col.startswith("glucose_rolling_")
    ]

    target_cols = [f"target_glucose_{horizon}min" for horizon in FORECAST_HORIZONS]

    keep_cols = [
        "patient_id",
        "split",
        "source_file",
        "timestamp",
        "glucose",
        "glucose_observed",
    ] + feature_cols + target_cols

    dataset = dataset[keep_cols].copy()

    before = len(dataset)

    # Require a full 2-hour history and both future targets.
    dataset = dataset.dropna(subset=feature_cols + target_cols).copy()

    after = len(dataset)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved CGM lag dataset to {OUT_PATH}")
    print(f"Rows before dropping incomplete windows: {before:,}")
    print(f"Rows after dropping incomplete windows:  {after:,}")
    print()
    print("Rows by split:")
    print(dataset["split"].value_counts())
    print()
    print("Rows by patient/split:")
    print(dataset.groupby(["patient_id", "split"]).size())
    print()
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {target_cols}")


if __name__ == "__main__":
    main()