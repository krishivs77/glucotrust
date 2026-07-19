from pathlib import Path

import pandas as pd


TIMELINE_PATH = Path("data/interim/timelines/cgm_timeline.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "cgm_lag_dataset.csv"

GRID_MINUTES = 5
LAG_MINUTES = list(range(0, 125, 5))  # 0, 5, 10, ..., 120
FORECAST_HORIZONS = [30, 60]


def load_timeline() -> pd.DataFrame:
    if not TIMELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TIMELINE_PATH}. "
            "Run src/data/build_cgm_timeline.py first."
        )

    df = pd.read_csv(TIMELINE_PATH)

    required_cols = {
        "patient_id",
        "split",
        "source_file",
        "timestamp",
        "glucose_observed",
        "glucose_causal",
        "glucose_state",
        "glucose_source_timestamp",
        "glucose_age_minutes",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns from CGM timeline: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["glucose_source_timestamp"] = pd.to_datetime(
        df["glucose_source_timestamp"],
        errors="coerce",
    )

    df["glucose_observed"] = pd.to_numeric(
        df["glucose_observed"],
        errors="coerce",
    )

    df["glucose_causal"] = pd.to_numeric(
        df["glucose_causal"],
        errors="coerce",
    )

    df["glucose_age_minutes"] = pd.to_numeric(
        df["glucose_age_minutes"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("CGM timeline contains invalid timestamps.")

    # Keep missing glucose rows. They are part of the complete time grid.
    df = df.sort_values(
        ["patient_id", "split", "timestamp"]
    ).reset_index(drop=True)

    return df


def validate_group_timeline(group: pd.DataFrame) -> None:
    if group["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found within a patient/split timeline."
        )

    timestamp_differences = group["timestamp"].diff().dropna()

    expected_difference = pd.Timedelta(minutes=GRID_MINUTES)

    invalid_spacing = timestamp_differences != expected_difference

    if invalid_spacing.any():
        bad_differences = timestamp_differences.loc[invalid_spacing].head()

        raise ValueError(
            "Timeline is not a complete five-minute grid. "
            f"Example invalid differences:\n{bad_differences}"
        )


def add_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()

    validate_group_timeline(group)

    group = group.set_index("timestamp", drop=False)

    observed_by_time = group["glucose_observed"]
    causal_by_time = group["glucose_causal"]

    # Lag 0 must be a real observation.
    group["glucose_lag_0min"] = observed_by_time.to_numpy()

    # Historical lags may use timestamp-safe, past-only causal values.
    for minutes in LAG_MINUTES:
        if minutes == 0:
            continue

        requested_timestamps = (
            group.index - pd.Timedelta(minutes=minutes)
        )

        group[f"glucose_lag_{minutes}min"] = (
            causal_by_time.reindex(requested_timestamps).to_numpy()
        )

    # Targets must be observed at the exact future timestamp.
    for horizon in FORECAST_HORIZONS:
        requested_timestamps = (
            group.index + pd.Timedelta(minutes=horizon)
        )

        group[f"target_glucose_{horizon}min"] = (
            observed_by_time.reindex(requested_timestamps).to_numpy()
        )

    # Changes are calculated from exact timestamp lags.
    group["glucose_change_30min"] = (
        group["glucose_lag_0min"]
        - group["glucose_lag_30min"]
    )

    group["glucose_change_60min"] = (
        group["glucose_lag_0min"]
        - group["glucose_lag_60min"]
    )

    group["glucose_change_120min"] = (
        group["glucose_lag_0min"]
        - group["glucose_lag_120min"]
    )

    # Rolling features use exact historical lag timestamps.
    #
    # A 30-minute window includes:
    # t, t-5, ..., t-30
    #
    # A 60-minute window includes:
    # t, t-5, ..., t-60
    rolling_30_columns = [
        f"glucose_lag_{minutes}min"
        for minutes in range(0, 35, 5)
    ]

    rolling_60_columns = [
        f"glucose_lag_{minutes}min"
        for minutes in range(0, 65, 5)
    ]

    group["glucose_rolling_mean_30min"] = group[
        rolling_30_columns
    ].mean(axis=1, skipna=False)

    group["glucose_rolling_std_30min"] = group[
        rolling_30_columns
    ].std(axis=1, skipna=False)

    group["glucose_rolling_mean_60min"] = group[
        rolling_60_columns
    ].mean(axis=1, skipna=False)

    group["glucose_rolling_std_60min"] = group[
        rolling_60_columns
    ].std(axis=1, skipna=False)

    return group.reset_index(drop=True)


def main() -> None:
    timeline = load_timeline()

    datasets = []

    grouped = timeline.groupby(
        ["patient_id", "split"],
        sort=True,
    )

    for (patient_id, split), group in grouped:
        print(
            "Building timestamp-safe lag features for "
            f"patient={patient_id}, split={split}"
        )

        datasets.append(add_lag_features(group))

    dataset = pd.concat(datasets, ignore_index=True)

    lag_cols = [
        f"glucose_lag_{minutes}min"
        for minutes in LAG_MINUTES
    ]

    change_cols = [
        "glucose_change_30min",
        "glucose_change_60min",
        "glucose_change_120min",
    ]

    rolling_cols = [
        "glucose_rolling_mean_30min",
        "glucose_rolling_std_30min",
        "glucose_rolling_mean_60min",
        "glucose_rolling_std_60min",
    ]

    feature_cols = lag_cols + change_cols + rolling_cols

    target_30_col = "target_glucose_30min"
    target_60_col = "target_glucose_60min"

    # A row is feature-eligible only when:
    # - current glucose is directly observed
    # - every required timestamp-safe feature is available
    dataset["eligible_features"] = (
        dataset["glucose_lag_0min"].notna()
        & dataset[feature_cols].notna().all(axis=1)
    )

    # Target eligibility is determined independently by horizon.
    dataset["eligible_target_30min"] = (
        dataset[target_30_col].notna()
    )

    dataset["eligible_target_60min"] = (
        dataset[target_60_col].notna()
    )

    dataset["eligible_30min"] = (
        dataset["eligible_features"]
        & dataset["eligible_target_30min"]
    )

    dataset["eligible_60min"] = (
        dataset["eligible_features"]
        & dataset["eligible_target_60min"]
    )

    keep_cols = [
        "patient_id",
        "split",
        "source_file",
        "timestamp",
        "glucose_observed",
        "glucose_causal",
        "glucose_state",
        "glucose_source_timestamp",
        "glucose_age_minutes",
    ]

    keep_cols += feature_cols

    keep_cols += [
        target_30_col,
        target_60_col,
        "eligible_features",
        "eligible_target_30min",
        "eligible_target_60min",
        "eligible_30min",
        "eligible_60min",
    ]

    dataset = dataset[keep_cols].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved CGM lag dataset to {OUT_PATH}")
    print(f"Total timeline rows: {len(dataset):,}")
    print(
        "Feature-eligible rows: "
        f"{dataset['eligible_features'].sum():,}"
    )
    print(
        "30-minute eligible rows: "
        f"{dataset['eligible_30min'].sum():,}"
    )
    print(
        "60-minute eligible rows: "
        f"{dataset['eligible_60min'].sum():,}"
    )
    print()

    print("Eligibility by split:")
    print(
        dataset.groupby("split")[
            [
                "eligible_features",
                "eligible_30min",
                "eligible_60min",
            ]
        ].sum()
    )

    print()
    print("Eligibility combinations:")
    print(
        dataset[
            ["eligible_30min", "eligible_60min"]
        ].value_counts()
    )

    print()
    print(f"Feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()