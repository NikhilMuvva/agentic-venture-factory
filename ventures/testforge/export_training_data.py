import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
APPROVED_DIR = PROJECT_DIR / "data" / "approved"
EXPORT_PATH = PROJECT_DIR / "data" / "approved_training.jsonl"


def export_training_data(
    approved_dir: Path = APPROVED_DIR,
    output_path: Path = EXPORT_PATH,
) -> int:
    """Export only records with explicit human approval and verified training rights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligible_records: list[dict[str, Any]] = []

    if approved_dir.exists():
        for path in sorted(approved_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as data_file:
                record = json.load(data_file)
            if record.get("human_approved") is True and record.get("training_rights_verified") is True:
                eligible_records.append(record)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in eligible_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    return len(eligible_records)


if __name__ == "__main__":
    count = export_training_data()
    print(f"Exported {count} rights-cleared human-approved question(s) to {EXPORT_PATH}.")
