from pathlib import Path
import xml.etree.ElementTree as ET


RAW_DIR = Path("data/raw")


def inspect_xml_file(xml_path: Path) -> None:
    print("=" * 100)
    print(f"File: {xml_path}")
    print("=" * 100)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Could not parse XML: {e}")
        return

    print(f"Root tag: {root.tag}")
    print(f"Root attributes: {root.attrib}")

    print("\nEvent streams:")
    for child in root:
        events = list(child)
        example = events[0].attrib if events else {}
        print(f"  - {child.tag}: {len(events)} events")
        if example:
            print(f"    example attributes: {list(example.keys())}")

    print()


def main() -> None:
    xml_files = sorted(RAW_DIR.rglob("*.xml"))

    if not xml_files:
        print(f"No XML files found under {RAW_DIR.resolve()}")
        return

    print(f"Found {len(xml_files)} XML file(s).")

    for xml_path in xml_files:
        inspect_xml_file(xml_path)


if __name__ == "__main__":
    main()