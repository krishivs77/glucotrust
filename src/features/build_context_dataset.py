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
            f"Missing {LAG_DATASET_PATH}. Run src/features/build_cgm_lag_dataset.py first."
        )

    df = pd.read_csv(LAG_DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


def load_event_table(name: str) -> pd.DataFrame:
    path = EVENTS_DIR / f"{name}.csv"

    if not path.exists():
        print(f"Warning: missing event table {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in ["ts", "ts_begin", "ts_end", "tbegin", "tend"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def floor_to_5min(series: pd.Series) -> pd.Series:
    return series.dt.floor("5min")


def add_rolling_event_sums(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
    prefix: str,
    timestamp_col: str = "ts",
) -> pd.DataFrame:
    timeline = timeline.copy()

    for window in WINDOW_MINUTES:
        timeline[f"{prefix}_last_{window}min"] = 0.0

    if events.empty or value_col not in events.columns or timestamp_col not in events.columns:
        return timeline

    events = events.copy()
    events = events.dropna(subset=[timestamp_col])
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce").fillna(0.0)
    events["timestamp"] = floor_to_5min(events[timestamp_col])

    grouped_events = (
        events.groupby(["patient_id", "split", "timestamp"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: f"{prefix}_at_time"})
    )

    timeline = timeline.merge(
        grouped_events,
        on=["patient_id", "split", "timestamp"],
        how="left",
    )

    timeline[f"{prefix}_at_time"] = timeline[f"{prefix}_at_time"].fillna(0.0)

    output_groups = []

    for (_patient_id, _split), group in timeline.groupby(["patient_id", "split"], sort=True):
        group = group.sort_values("timestamp").copy()

        for window in WINDOW_MINUTES:
            # Number of 5-minute bins in the window, including the current timestamp.
            window_steps = (window // 5) + 1

            group[f"{prefix}_last_{window}min"] = (
                group[f"{prefix}_at_time"]
                .rolling(window=window_steps, min_periods=1)
                .sum()
            )

        output_groups.append(group)

    timeline = pd.concat(output_groups, ignore_index=True)
    timeline = timeline.drop(columns=[f"{prefix}_at_time"])

    return timeline


def add_event_count_sums(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    timestamp_col: str = "ts",
) -> pd.DataFrame:
    timeline = timeline.copy()

    for window in WINDOW_MINUTES:
        timeline[f"{prefix}_count_last_{window}min"] = 0.0

    if events.empty or timestamp_col not in events.columns:
        return timeline

    events = events.copy()
    events = events.dropna(subset=[timestamp_col])
    events["timestamp"] = floor_to_5min(events[timestamp_col])
    events["event_count"] = 1.0

    grouped_events = (
        events.groupby(["patient_id", "split", "timestamp"], as_index=False)["event_count"]
        .sum()
        .rename(columns={"event_count": f"{prefix}_count_at_time"})
    )

    timeline = timeline.merge(
        grouped_events,
        on=["patient_id", "split", "timestamp"],
        how="left",
    )

    timeline[f"{prefix}_count_at_time"] = timeline[f"{prefix}_count_at_time"].fillna(0.0)

    output_groups = []

    for (_patient_id, _split), group in timeline.groupby(["patient_id", "split"], sort=True):
        group = group.sort_values("timestamp").copy()

        for window in WINDOW_MINUTES:
            window_steps = (window // 5) + 1

            group[f"{prefix}_count_last_{window}min"] = (
                group[f"{prefix}_count_at_time"]
                .rolling(window=window_steps, min_periods=1)
                .sum()
            )

        output_groups.append(group)

    timeline = pd.concat(output_groups, ignore_index=True)
    timeline = timeline.drop(columns=[f"{prefix}_count_at_time"])

    return timeline


def add_time_since_event(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    timestamp_col: str = "ts",
) -> pd.DataFrame:
    timeline = timeline.copy()

    output_groups = []

    for (patient_id, split), group in timeline.groupby(["patient_id", "split"], sort=True):
        group = group.sort_values("timestamp").copy()

        event_group = events[
            (events["patient_id"].astype(str) == str(patient_id))
            & (events["split"].astype(str) == str(split))
        ].copy()

        if event_group.empty or timestamp_col not in event_group.columns:
            group[f"time_since_last_{prefix}_min"] = MAX_TIME_SINCE_MINUTES
            group[f"has_prior_{prefix}"] = 0
            output_groups.append(group)
            continue

        event_group = event_group.dropna(subset=[timestamp_col])
        event_group = event_group.sort_values(timestamp_col)
        event_group = event_group[[timestamp_col]].rename(columns={timestamp_col: f"last_{prefix}_time"})

        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            event_group,
            left_on="timestamp",
            right_on=f"last_{prefix}_time",
            direction="backward",
        )

        time_since = (
            merged["timestamp"] - merged[f"last_{prefix}_time"]
        ).dt.total_seconds() / 60.0

        merged[f"has_prior_{prefix}"] = time_since.notna().astype(int)
        merged[f"time_since_last_{prefix}_min"] = (
            time_since.fillna(MAX_TIME_SINCE_MINUTES)
            .clip(lower=0, upper=MAX_TIME_SINCE_MINUTES)
        )

        merged = merged.drop(columns=[f"last_{prefix}_time"])
        output_groups.append(merged)

    return pd.concat(output_groups, ignore_index=True)


def add_current_basal_rate(timeline: pd.DataFrame, basal: pd.DataFrame) -> pd.DataFrame:
    timeline = timeline.copy()
    output_groups = []

    if basal.empty or "ts" not in basal.columns or "value" not in basal.columns:
        timeline["basal_rate_current"] = 0.0
        timeline["basal_rate_known"] = 0
        return timeline

    basal = basal.copy()
    basal["value"] = pd.to_numeric(basal["value"], errors="coerce")
    basal = basal.dropna(subset=["ts", "value"])

    for (patient_id, split), group in timeline.groupby(["patient_id", "split"], sort=True):
        group = group.sort_values("timestamp").copy()

        basal_group = basal[
            (basal["patient_id"].astype(str) == str(patient_id))
            & (basal["split"].astype(str) == str(split))
        ].copy()

        if basal_group.empty:
            group["basal_rate_current"] = 0.0
            group["basal_rate_known"] = 0
            output_groups.append(group)
            continue

        basal_group = basal_group.sort_values("ts")
        basal_group = basal_group[["ts", "value"]].rename(
            columns={"ts": "basal_time", "value": "basal_rate_current"}
        )

        merged = pd.merge_asof(
            group,
            basal_group,
            left_on="timestamp",
            right_on="basal_time",
            direction="backward",
        )

        merged["basal_rate_known"] = merged["basal_rate_current"].notna().astype(int)
        merged["basal_rate_current"] = merged["basal_rate_current"].fillna(0.0)
        merged = merged.drop(columns=["basal_time"])

        output_groups.append(merged)

    return pd.concat(output_groups, ignore_index=True)


def get_context_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "carbs_",
        "meal_",
        "bolus_",
        "bolus_carb_input_",
        "time_since_",
        "has_prior_",
        "basal_",
    )

    return [col for col in df.columns if col.startswith(prefixes)]


def main() -> None:
    dataset = load_lag_dataset()

    print(f"Loaded lag dataset: {len(dataset):,} rows")

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

    print("Adding meal carbohydrate features...")
    context = add_rolling_event_sums(
        timeline=context,
        events=meals,
        value_col="carbs",
        prefix="carbs",
        timestamp_col="ts",
    )
    context = add_event_count_sums(
        timeline=context,
        events=meals,
        prefix="meal",
        timestamp_col="ts",
    )
    context = add_time_since_event(
        timeline=context,
        events=meals,
        prefix="meal",
        timestamp_col="ts",
    )

    print("Adding bolus insulin features...")
    context = add_rolling_event_sums(
        timeline=context,
        events=bolus,
        value_col="dose",
        prefix="bolus_units",
        timestamp_col="ts_begin",
    )
    context = add_rolling_event_sums(
        timeline=context,
        events=bolus,
        value_col="bwz_carb_input",
        prefix="bolus_carb_input",
        timestamp_col="ts_begin",
    )
    context = add_event_count_sums(
        timeline=context,
        events=bolus,
        prefix="bolus",
        timestamp_col="ts_begin",
    )
    context = add_time_since_event(
        timeline=context,
        events=bolus,
        prefix="bolus",
        timestamp_col="ts_begin",
    )

    print("Adding basal insulin features...")
    context = add_current_basal_rate(context, basal)

    context_feature_cols = get_context_feature_columns(context)

    merged = dataset.merge(
        context[timeline_cols + context_feature_cols],
        on=timeline_cols,
        how="left",
    )

    # Fill any remaining context missingness conservatively.
    for col in context_feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved context dataset to {OUT_PATH}")
    print(f"Rows: {len(merged):,}")
    print(f"Original columns: {len(dataset.columns)}")
    print(f"Context feature columns: {len(context_feature_cols)}")
    print(f"Total columns: {len(merged.columns)}")
    print()
    print("Context features:")
    for col in context_feature_cols:
        print(f"  - {col}")

    print()
    print("Context feature nonzero counts:")
    nonzero_counts = {}
    for col in context_feature_cols:
        nonzero_counts[col] = int((merged[col] != 0).sum())

    nonzero_df = pd.DataFrame(
        {
            "feature": list(nonzero_counts.keys()),
            "nonzero_count": list(nonzero_counts.values()),
        }
    ).sort_values("nonzero_count", ascending=False)

    print(nonzero_df.to_string(index=False))


if __name__ == "__main__":
    main()