from pathlib import Path

import pandas as pd


EVENTS_DIR = Path("data/interim/events")
OUT_DIR = Path("data/interim/timelines")
OUT_PATH = OUT_DIR / "cgm_timeline.csv"


def load_glucose_events() -> pd.DataFrame:
    glucose_path = EVENTS_DIR / "glucose_level.csv"

    if not glucose_path.exists():
        raise FileNotFoundError(
            f"Missing {glucose_path}. Run src/data/parse_xml_events.py first."
        )

    df = pd.read_csv(glucose_path)

    required_cols = {"patient_id", "split", "ts", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns from glucose file: {missing}")

    df = df[["patient_id", "split", "source_file", "ts", "value"]].copy()
    df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "glucose"])
    df = df.drop(columns=["ts", "value"])

    return df


def resample_patient_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    patient_id = df["patient_id"].iloc[0]
    split = df["split"].iloc[0]
    source_file = df["source_file"].iloc[0]

    # Average duplicate readings within the same 5-minute bin.
    timeline = (
        df.set_index("timestamp")
        .resample("5min")["glucose"]
        .mean()
        .to_frame()
    )

    timeline["patient_id"] = patient_id
    timeline["split"] = split
    timeline["source_file"] = source_file

    # Track whether this timestamp had an observed CGM value before interpolation.
    timeline["glucose_observed"] = timeline["glucose"].notna().astype(int)

    # Fill short CGM gaps only. Limit 3 = up to 15 minutes.
    timeline["glucose"] = timeline["glucose"].interpolate(
        method="time",
        limit=3,
        limit_direction="both",
    )

    timeline = timeline.reset_index().rename(columns={"timestamp": "timestamp"})

    return timeline


def main() -> None:
    glucose = load_glucose_events()

    timelines = []

    grouped = glucose.groupby(["patient_id", "split"], sort=True)

    for (patient_id, split), group in grouped:
        print(f"Building CGM timeline for patient={patient_id}, split={split}")
        timelines.append(resample_patient_split(group))

    timeline = pd.concat(timelines, ignore_index=True)

    # Keep rows where glucose is available after short-gap interpolation.
    before = len(timeline)
    timeline = timeline.dropna(subset=["glucose"]).copy()
    after = len(timeline)

    # Nice ordering.
    timeline = timeline[
        [
            "patient_id",
            "split",
            "source_file",
            "timestamp",
            "glucose",
            "glucose_observed",
        ]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved CGM timeline to {OUT_PATH}")
    print(f"Rows before dropping long gaps: {before:,}")
    print(f"Rows after dropping long gaps:  {after:,}")
    print()
    print("Rows by split:")
    print(timeline["split"].value_counts())
    print()
    print("Rows by patient/split:")
    print(timeline.groupby(["patient_id", "split"]).size())


if __name__ == "__main__":
    main()