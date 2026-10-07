import json
import os
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
APPROVED_PATH = DATA_DIR / "approved_questions.json"
REJECTED_PATH = DATA_DIR / "rejected_questions.json"
RUNS_PATH = DATA_DIR / "runs.json"
RUNS_PATH = DATA_DIR / "runs.json"
APPROVED_DIR = DATA_DIR / "approved"
REJECTED_DIR = DATA_DIR / "rejected"
VISUALS_DIR = DATA_DIR / "visuals"
RUNS_DIR = DATA_DIR / "runs"


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (APPROVED_PATH, REJECTED_PATH, RUNS_PATH):
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")


def ensure_pipeline_directories() -> None:
    for directory in (APPROVED_DIR, REJECTED_DIR, VISUALS_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as data_file:
        json.dump(record, data_file, indent=2, ensure_ascii=False)
        data_file.write("\n")
    os.replace(temporary_path, path)


def read_records(path: Path) -> list[dict[str, Any]]:
    ensure_data_files()
    with path.open("r", encoding="utf-8") as data_file:
        records = json.load(data_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path.name}.")
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_data_files()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as data_file:
        json.dump(records, data_file, indent=2, ensure_ascii=False)
        data_file.write("\n")
    os.replace(temporary_path, path)


def append_record(path: Path, record: dict[str, Any]) -> None:
    records = read_records(path)
    records.append(record)
    write_records(path, records)
