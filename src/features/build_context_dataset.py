from pathlib import Path

import numpy as np
import pandas as pd


LAG_DATASET_PATH = Path("data/processed/cgm_lag_dataset.csv")
EVENTS_DIR = Path("data/interim/events")
OUT_PATH = Path("data/processed/cgm_context_dataset.csv")

WINDOW_MINUTES = [30, 60, 120, 180]
MAX_TIME_SINCE_MINUTES = 24 * 60


def load_lag_dataset() -> pd.DataFrame:
    if not LAG_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LAG_DATASET_PATH}. "
            "Run src/features/build_cgm_lag_dataset.py first."
        )

    df = pd.read_csv(LAG_DATASET_PATH)

    required_cols = {
        "patient_id",
        "split",
        "source_file",
        "timestamp",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns from lag dataset: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Lag dataset contains invalid timestamps.")

    duplicate_keys = df.duplicated(
        subset=["patient_id", "split", "timestamp"]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Lag dataset contains duplicate patient/split/timestamp rows."
        )

    return df.sort_values(
        ["patient_id", "split", "timestamp"]
    ).reset_index(drop=True)


def load_event_table(name: str) -> pd.DataFrame:
    path = EVENTS_DIR / f"{name}.csv"

    if not path.exists():
        print(
            f"Warning: missing event table {path}. "
            "Returning an empty DataFrame."
        )
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in [
        "ts",
        "ts_begin",
        "ts_end",
        "tbegin",
        "tend",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

    return df


def filter_events_for_group(
    events: pd.DataFrame,
    patient_id: object,
    split: object,
    timestamp_col: str,
) -> pd.DataFrame:
    if (
        events.empty
        or timestamp_col not in events.columns
        or "patient_id" not in events.columns
        or "split" not in events.columns
    ):
        return pd.DataFrame()

    event_group = events[
        (events["patient_id"].astype(str) == str(patient_id))
        & (events["split"].astype(str) == str(split))
    ].copy()

    event_group = event_group.dropna(
        subset=[timestamp_col]
    )

    return event_group.sort_values(timestamp_col)


def calculate_window_counts(
    timeline_times: np.ndarray,
    event_times: np.ndarray,
    window_minutes: int,
) -> np.ndarray:
    window_delta = np.timedelta64(
        window_minutes,
        "m",
    )

    window_starts = timeline_times - window_delta

    # Each feature window is inclusive:
    # [prediction time - window, prediction time]
    left_positions = np.searchsorted(
        event_times,
        window_starts,
        side="left",
    )

    right_positions = np.searchsorted(
        event_times,
        timeline_times,
        side="right",
    )

    return (
        right_positions - left_positions
    ).astype(float)


def calculate_window_sums(
    timeline_times: np.ndarray,
    event_times: np.ndarray,
    event_values: np.ndarray,
    window_minutes: int,
) -> np.ndarray:
    window_delta = np.timedelta64(
        window_minutes,
        "m",
    )

    window_starts = timeline_times - window_delta

    left_positions = np.searchsorted(
        event_times,
        window_starts,
        side="left",
    )

    right_positions = np.searchsorted(
        event_times,
        timeline_times,
        side="right",
    )

    cumulative_values = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(event_values),
        ]
    )

    return (
        cumulative_values[right_positions]
        - cumulative_values[left_positions]
    )


def add_windowed_event_features(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    timestamp_col: str,
    value_features: dict[str, str],
    count_prefix: str,
) -> pd.DataFrame:
    """
    Add event sums and counts using actual elapsed-time intervals.

    value_features maps:
        event value column -> output feature prefix

    Example:
        {"carbs": "carbs"}
    """

    timeline = timeline.copy()
    output_groups = []

    for prefix in value_features.values():
        for window in WINDOW_MINUTES:
            timeline[f"{prefix}_last_{window}min"] = 0.0

    for window in WINDOW_MINUTES:
        timeline[
            f"{count_prefix}_count_last_{window}min"
        ] = 0.0

    for (patient_id, split), group in timeline.groupby(
        ["patient_id", "split"],
        sort=True,
    ):
        group = group.sort_values("timestamp").copy()

        event_group = filter_events_for_group(
            events=events,
            patient_id=patient_id,
            split=split,
            timestamp_col=timestamp_col,
        )

        if event_group.empty:
            output_groups.append(group)
            continue

        timeline_times = group[
            "timestamp"
        ].to_numpy(dtype="datetime64[ns]")

        event_times = event_group[
            timestamp_col
        ].to_numpy(dtype="datetime64[ns]")

        for window in WINDOW_MINUTES:
            group[
                f"{count_prefix}_count_last_{window}min"
            ] = calculate_window_counts(
                timeline_times=timeline_times,
                event_times=event_times,
                window_minutes=window,
            )

        for value_col, prefix in value_features.items():
            if value_col not in event_group.columns:
                event_values = np.zeros(
                    len(event_group),
                    dtype=float,
                )
            else:
                event_values = pd.to_numeric(
                    event_group[value_col],
                    errors="coerce",
                ).fillna(0.0).to_numpy(dtype=float)

            for window in WINDOW_MINUTES:
                group[
                    f"{prefix}_last_{window}min"
                ] = calculate_window_sums(
                    timeline_times=timeline_times,
                    event_times=event_times,
                    event_values=event_values,
                    window_minutes=window,
                )

        output_groups.append(group)

    return pd.concat(
        output_groups,
        ignore_index=True,
    )


def add_time_since_event(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    timestamp_col: str,
) -> pd.DataFrame:
    timeline = timeline.copy()
    output_groups = []

    for (patient_id, split), group in timeline.groupby(
        ["patient_id", "split"],
        sort=True,
    ):
        group = group.sort_values("timestamp").copy()

        event_group = filter_events_for_group(
            events=events,
            patient_id=patient_id,
            split=split,
            timestamp_col=timestamp_col,
        )

        if event_group.empty:
            group[
                f"time_since_last_{prefix}_min"
            ] = float(MAX_TIME_SINCE_MINUTES)

            group[f"has_prior_{prefix}"] = 0

            output_groups.append(group)
            continue

        event_times = (
            event_group[[timestamp_col]]
            .drop_duplicates()
            .rename(
                columns={
                    timestamp_col: f"last_{prefix}_time"
                }
            )
            .sort_values(f"last_{prefix}_time")
        )

        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            event_times,
            left_on="timestamp",
            right_on=f"last_{prefix}_time",
            direction="backward",
            allow_exact_matches=True,
        )

        time_since = (
            merged["timestamp"]
            - merged[f"last_{prefix}_time"]
        ).dt.total_seconds() / 60.0

        merged[f"has_prior_{prefix}"] = (
            time_since.notna().astype(int)
        )

        merged[
            f"time_since_last_{prefix}_min"
        ] = (
            time_since
            .fillna(MAX_TIME_SINCE_MINUTES)
            .clip(
                lower=0,
                upper=MAX_TIME_SINCE_MINUTES,
            )
        )

        merged = merged.drop(
            columns=[f"last_{prefix}_time"]
        )

        output_groups.append(merged)

    return pd.concat(
        output_groups,
        ignore_index=True,
    )


def add_current_basal_rate(
    timeline: pd.DataFrame,
    basal: pd.DataFrame,
) -> pd.DataFrame:
    timeline = timeline.copy()
    output_groups = []

    required_columns = {
        "patient_id",
        "split",
        "ts",
        "value",
    }

    if basal.empty or not required_columns.issubset(
        basal.columns
    ):
        timeline["basal_rate_current"] = 0.0
        timeline["basal_rate_known"] = 0
        return timeline

    basal = basal.copy()

    basal["value"] = pd.to_numeric(
        basal["value"],
        errors="coerce",
    )

    basal = basal.dropna(
        subset=["ts", "value"]
    )

    for (patient_id, split), group in timeline.groupby(
        ["patient_id", "split"],
        sort=True,
    ):
        group = group.sort_values("timestamp").copy()

        basal_group = filter_events_for_group(
            events=basal,
            patient_id=patient_id,
            split=split,
            timestamp_col="ts",
        )

        if basal_group.empty:
            group["basal_rate_current"] = 0.0
            group["basal_rate_known"] = 0
            output_groups.append(group)
            continue

        basal_group = (
            basal_group[["ts", "value"]]
            .rename(
                columns={
                    "ts": "basal_time",
                    "value": "basal_rate_current",
                }
            )
            .sort_values("basal_time")
            .drop_duplicates(
                subset=["basal_time"],
                keep="last",
            )
        )

        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            basal_group,
            left_on="timestamp",
            right_on="basal_time",
            direction="backward",
            allow_exact_matches=True,
        )

        merged["basal_rate_known"] = (
            merged["basal_rate_current"]
            .notna()
            .astype(int)
        )

        merged["basal_rate_current"] = (
            merged["basal_rate_current"]
            .fillna(0.0)
        )

        merged = merged.drop(
            columns=["basal_time"]
        )

        output_groups.append(merged)

    return pd.concat(
        output_groups,
        ignore_index=True,
    )


def get_context_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    prefixes = (
        "carbs_",
        "meal_",
        "bolus_",
        "bolus_carb_input_",
        "time_since_",
        "has_prior_",
        "basal_",
    )

    return [
        col
        for col in df.columns
        if col.startswith(prefixes)
    ]


def validate_context_features(
    df: pd.DataFrame,
    context_feature_cols: list[str],
) -> None:
    if len(context_feature_cols) != 26:
        raise ValueError(
            "Expected 26 context features, "
            f"but found {len(context_feature_cols)}."
        )

    duplicate_keys = df.duplicated(
        subset=["patient_id", "split", "timestamp"]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Context dataset contains duplicate "
            "patient/split/timestamp rows."
        )

    missing_values = df[
        context_feature_cols
    ].isna().sum().sum()

    if missing_values:
        raise ValueError(
            "Context features contain "
            f"{missing_values:,} missing values."
        )

    negative_sum_or_count_cols = [
        col
        for col in context_feature_cols
        if (
            "_last_" in col
            and (
                col.startswith("carbs_")
                or col.startswith("meal_")
                or col.startswith("bolus_")
            )
        )
    ]

    if (
        df[negative_sum_or_count_cols] < 0
    ).any().any():
        raise ValueError(
            "Event sums or counts contain negative values."
        )

    for prefix in ["meal", "bolus"]:
        indicator_col = f"has_prior_{prefix}"
        time_col = f"time_since_last_{prefix}_min"

        if not df[indicator_col].isin([0, 1]).all():
            raise ValueError(
                f"{indicator_col} contains values other than 0 and 1."
            )

        if not df[time_col].between(
            0,
            MAX_TIME_SINCE_MINUTES,
        ).all():
            raise ValueError(
                f"{time_col} is outside its allowed range."
            )

    if not df["basal_rate_known"].isin([0, 1]).all():
        raise ValueError(
            "basal_rate_known contains values other than 0 and 1."
        )


def main() -> None:
    dataset = load_lag_dataset()

    print(
        f"Loaded timestamp-safe lag dataset: "
        f"{len(dataset):,} rows"
    )

    meals = load_event_table("meal")
    bolus = load_event_table("bolus")
    basal = load_event_table("basal")

    timeline_cols = [
        "patient_id",
        "split",
        "source_file",
        "timestamp",
    ]

    context = dataset[timeline_cols].copy()

    print("Adding timestamp-based meal features...")

    context = add_windowed_event_features(
        timeline=context,
        events=meals,
        timestamp_col="ts",
        value_features={
            "carbs": "carbs",
        },
        count_prefix="meal",
    )

    context = add_time_since_event(
        timeline=context,
        events=meals,
        prefix="meal",
        timestamp_col="ts",
    )

    print("Adding timestamp-based bolus features...")

    context = add_windowed_event_features(
        timeline=context,
        events=bolus,
        timestamp_col="ts_begin",
        value_features={
            "dose": "bolus_units",
            "bwz_carb_input": "bolus_carb_input",
        },
        count_prefix="bolus",
    )

    context = add_time_since_event(
        timeline=context,
        events=bolus,
        prefix="bolus",
        timestamp_col="ts_begin",
    )

    print("Adding timestamp-based basal features...")

    context = add_current_basal_rate(
        timeline=context,
        basal=basal,
    )

    context_feature_cols = get_context_feature_columns(
        context
    )

    merged = dataset.merge(
        context[
            timeline_cols + context_feature_cols
        ],
        on=timeline_cols,
        how="left",
        validate="one_to_one",
    )

    for col in context_feature_cols:
        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce",
        )

    validate_context_features(
        df=merged,
        context_feature_cols=context_feature_cols,
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print(f"Saved context dataset to {OUT_PATH}")
    print(f"Rows: {len(merged):,}")
    print(f"Original columns: {len(dataset.columns)}")
    print(
        "Context feature columns: "
        f"{len(context_feature_cols)}"
    )
    print(f"Total columns: {len(merged.columns)}")

    print()
    print("Context features:")

    for col in context_feature_cols:
        print(f"  - {col}")

    print()
    print("Context feature nonzero counts:")

    nonzero_counts = {
        col: int((merged[col] != 0).sum())
        for col in context_feature_cols
    }

    nonzero_df = pd.DataFrame(
        {
            "feature": list(nonzero_counts.keys()),
            "nonzero_count": list(
                nonzero_counts.values()
            ),
        }
    ).sort_values(
        "nonzero_count",
        ascending=False,
    )

    print(
        nonzero_df.to_string(index=False)
    )

    print()
    print("Context quality-control checks passed.")


if __name__ == "__main__":
    main()