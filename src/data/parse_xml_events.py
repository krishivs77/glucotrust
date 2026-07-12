from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


RAW_DIR = Path("data/raw/OhioT1DM")
OUT_DIR = Path("data/interim/events")


def infer_split(xml_path: Path) -> str:
    parts = [part.lower() for part in xml_path.parts]
    file_name = xml_path.name.lower()

    if "train" in parts or "training" in file_name:
        return "training"

    if "test" in parts or "testing" in file_name:
        return "testing"

    return "unknown"


def infer_cohort_year(xml_path: Path) -> str:
    for part in xml_path.parts:
        if part in {"2018", "2020"}:
            return part

    return "unknown"


def parse_timestamp(value):
    if value is None:
        return pd.NaT

    return pd.to_datetime(value, format="%d-%m-%Y %H:%M:%S", errors="coerce")


def parse_numeric(value):
    if value is None:
        return pd.NA

    try:
        return float(value)
    except ValueError:
        return value


def parse_xml_file(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cohort_year = infer_cohort_year(xml_path)
    patient_id = root.attrib.get("id")
    weight = parse_numeric(root.attrib.get("weight"))
    insulin_type = root.attrib.get("insulin_type")
    split = infer_split(xml_path)

    rows = []

    for stream in root:
        stream_name = stream.tag

        for event_idx, event in enumerate(stream):
            row = {
                "source_file": xml_path.name,
                "source_path": str(xml_path),
                "cohort_year": cohort_year,
                "patient_id": patient_id,
                "split": split,
                "weight": weight,
                "insulin_type": insulin_type,
                "stream": stream_name,
                "event_idx": event_idx,
            }

            for key, value in event.attrib.items():
                if key in {"ts", "ts_begin", "ts_end", "tbegin", "tend"}:
                    row[key] = parse_timestamp(value)
                else:
                    row[key] = parse_numeric(value)

            rows.append(row)

    return rows


def main() -> None:
    xml_files = sorted(RAW_DIR.rglob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {RAW_DIR.resolve()}")

    all_rows = []

    for xml_path in xml_files:
        print(f"Parsing {xml_path}")
        all_rows.extend(parse_xml_file(xml_path))

    events = pd.DataFrame(all_rows)

    if events.empty:
        raise ValueError("No events were parsed from XML files.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stream_names = sorted(events["stream"].dropna().unique())

    print()
    print(f"Parsed {len(events):,} total events across {len(stream_names)} streams.")
    print()

    for stream_name in stream_names:
        stream_df = events[events["stream"] == stream_name].copy()

        stream_df = stream_df.dropna(axis=1, how="all")

        out_path = OUT_DIR / f"{stream_name}.csv"
        stream_df.to_csv(out_path, index=False)

        print(f"Saved {len(stream_df):,} rows -> {out_path}")

    print()
    print(f"Saved event tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
