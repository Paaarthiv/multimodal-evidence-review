"""CSV reading/writing and image-path helpers.

Uses the stdlib csv module with QUOTE_ALL so the output exactly matches the
quoting style of the provided sample_claims.csv (every field quoted).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from . import schema


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_output_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write rows in the exact required column order, every field quoted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=schema.OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            # Ensure all columns exist
            writer.writerow({c: r.get(c, "") for c in schema.OUTPUT_COLUMNS})


def split_image_paths(image_paths: str) -> List[str]:
    """Split the semicolon-separated image_paths field into individual paths."""
    return [p.strip() for p in (image_paths or "").split(";") if p.strip()]


def image_id_from_path(p: str) -> str:
    """The image ID is the filename without extension, e.g. 'img_1'."""
    return Path(p).stem


def resolve_image_path(rel_path: str, dataset_dir: Path) -> Path:
    """Resolve a CSV image path (e.g. 'images/test/case_001/img_1.jpg')
    against the dataset directory."""
    return (dataset_dir / rel_path).resolve()
