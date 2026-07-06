"""Import PFF QB zip exports into data/raw/pff/<season>/.

Usage:
    python3 scripts/import_pff_qb_zips.py /path/to/pff_zips

The importer expects files named QB_YYYY.zip. Each zip should contain the PFF
family CSVs named like passing-grades__QB__YYYY.csv. Existing season files are
replaced so an import mirrors the source zips exactly.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RAW_PFF = REPO / "data" / "raw" / "pff"
ZIP_RE = re.compile(r"QB_(\d{4})\.zip$")


def import_zip(path: Path) -> tuple[int, list[str]]:
    match = ZIP_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected zip name: {path.name}")

    season = int(match.group(1))
    out_dir = RAW_PFF / str(season)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for member in sorted(zf.infolist(), key=lambda item: item.filename):
            name = os.path.basename(member.filename)
            if not name or not name.endswith(".csv"):
                continue
            if not name.endswith(f"__QB__{season}.csv"):
                raise ValueError(f"{path.name} contains unexpected CSV: {member.filename}")
            target = out_dir / name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)

    return season, extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_dir", help="Directory containing QB_YYYY.zip files")
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).expanduser().resolve()
    paths = sorted(zip_dir.glob("QB_*.zip"))
    if not paths:
        raise SystemExit(f"No QB_*.zip files found in {zip_dir}")

    for path in paths:
        season, files = import_zip(path)
        print(f"{season}: imported {len(files)} file(s)")
        for name in files:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
