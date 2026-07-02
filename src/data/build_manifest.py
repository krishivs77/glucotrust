from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/xml_manifest.csv")


TIMESTAMP_KEYS = ["ts", "ts_begin", "ts_end", "tbegin", "tend"]


def infer_split(file_name: str) -> str:
    lower = file_name.lower()
    if "training" in lower:
        return "training"
    if "testing" in lower:
        return "testing"
    return "unknown"


def parse_timestamp(value: str):
    if value is None:
        return pd.NaT

    # OhioT1DM timestamps look like: 29-10-2021 00:01:00
    return pd.to_datetime(value, format="%d-%m-%Y %H:%M:%S", errors="coerce")


def collect_file_time_bounds(root) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = []

    for stream in root:
        for event in stream:
            for key in TIMESTAMP_KEYS:
                if key in event.attrib:
                    ts = parse_timestamp(event.attrib.get(key))
                    if pd.notna(ts):
                        timestamps.append(ts)

    if not timestamps:
        return pd.NaT, pd.NaT

    return min(timestamps), max(timestamps)


def inspect_xml_file(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    patient_id = root.attrib.get("id")
    weight = root.attrib.get("weight")
    insulin_type = root.attrib.get("insulin_type")

    stream_counts = {child.tag: len(list(child)) for child in root}

    start_time, end_time = collect_file_time_bounds(root)

    if pd.notna(start_time) and pd.notna(end_time):
        duration_days = (end_time - start_time).total_seconds() / (60 * 60 * 24)
    else:
        duration_days = None

    row = {
        "file_name": xml_path.name,
        "file_path": str(xml_path),
        "patient_id": patient_id,
        "split": infer_split(xml_path.name),
        "weight": weight,
        "insulin_type": insulin_type,
        "start_time": start_time,
        "end_time": end_time,
        "duration_days": duration_days,
    }

    # Add one count column per stream.
    for stream_name, count in stream_counts.items():
        row[f"{stream_name}_count"] = count

    # Convenience flag for wearable data.
    wearable_streams = [
        "basis_heart_rate",
        "basis_gsr",
        "basis_skin_temperature",
        "basis_air_temperature",
        "basis_steps",
        "basis_sleep",
    ]
    row["has_wearable_streams"] = any(stream_counts.get(name, 0) > 0 for name in wearable_streams)

    return row


def main() -> None:
    xml_files = sorted(RAW_DIR.rglob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {RAW_DIR.resolve()}")

    rows = []
    for xml_path in xml_files:
        print(f"Inspecting {xml_path}")
        rows.append(inspect_xml_file(xml_path))

    manifest = pd.DataFrame(rows)

    # Nice column ordering.
    base_cols = [
        "file_name",
        "file_path",
        "patient_id",
        "split",
        "weight",
        "insulin_type",
        "start_time",
        "end_time",
        "duration_days",
        "has_wearable_streams",
    ]
    other_cols = [c for c in manifest.columns if c not in base_cols]
    manifest = manifest[base_cols + sorted(other_cols)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved manifest to {OUT_PATH}")
    print()
    print("Summary:")
    print(f"  XML files: {len(manifest)}")
    print(f"  Patients: {manifest['patient_id'].nunique()}")
    print(f"  Splits: {manifest['split'].value_counts().to_dict()}")
    print()
    print(manifest[["file_name", "patient_id", "split", "duration_days", "glucose_level_count"]])


if __name__ == "__main__":
    main()