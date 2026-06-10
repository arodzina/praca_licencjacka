#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scal_wyniki_z_danymi.py

Merges scoreboard columns (per-set points) and derived statistics
into a processed match data CSV by matching game_id.

Usage:
    python scraper/scal_wyniki_z_danymi.py
    python scraper/scal_wyniki_z_danymi.py --help
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCOREBOARD_FIELDS = [
    "A_set1_points",
    "B_set1_points",
    "A_set2_points",
    "B_set2_points",
    "A_set3_points",
    "B_set3_points",
    "A_set4_points",
    "B_set4_points",
    "A_set5_points",
    "B_set5_points",
    "A_scoreboard_points",
    "B_scoreboard_points",
]

DERIVED_FIELDS = [
    "match_total_points_from_scoreboard",
    "A_scoreboard_pts_per_set_from_scoreboard",
    "B_scoreboard_pts_per_set_from_scoreboard",
    "diff_scoreboard_pts_per_set_from_scoreboard",
]


def detect_delimiter(path: Path) -> str:
    """Detect whether a CSV file uses semicolon or comma as delimiter."""
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except csv.Error:
        return ";"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    delimiter = detect_delimiter(path)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames, delimiter


def to_number(value: str) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_scoreboard_row(row: dict[str, str]) -> dict[str, str]:
    """Compute derived scoreboard statistics from raw scoreboard fields."""
    out = {field: row.get(field, "") for field in SCOREBOARD_FIELDS}

    a_sb = to_number(out["A_scoreboard_points"])
    b_sb = to_number(out["B_scoreboard_points"])
    sety_a = to_number(row.get("sety_A", ""))
    sety_b = to_number(row.get("sety_B", ""))
    sets_sum = (sety_a or 0) + (sety_b or 0)

    out["match_total_points_from_scoreboard"] = (
        int(a_sb + b_sb) if a_sb is not None and b_sb is not None else ""
    )
    if sets_sum:
        out["A_scoreboard_pts_per_set_from_scoreboard"] = (
            a_sb / sets_sum if a_sb is not None else ""
        )
        out["B_scoreboard_pts_per_set_from_scoreboard"] = (
            b_sb / sets_sum if b_sb is not None else ""
        )
        if a_sb is not None and b_sb is not None:
            out["diff_scoreboard_pts_per_set_from_scoreboard"] = (a_sb - b_sb) / sets_sum
        else:
            out["diff_scoreboard_pts_per_set_from_scoreboard"] = ""
    else:
        out["A_scoreboard_pts_per_set_from_scoreboard"] = ""
        out["B_scoreboard_pts_per_set_from_scoreboard"] = ""
        out["diff_scoreboard_pts_per_set_from_scoreboard"] = ""

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge scoreboard columns into processed match data using game_id."
    )
    parser.add_argument(
        "--processed",
        default="data/processed_data_with_playoff.csv",
        help="Processed CSV to enrich.",
    )
    parser.add_argument(
        "--scoreboard-source",
        default="tauron_liga_statystyki2.csv",
        help="CSV containing scoreboard columns and game_id.",
    )
    parser.add_argument(
        "--output",
        default="data/processed_data_with_playoff_scoreboard.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    processed_path = Path(args.processed)
    scoreboard_path = Path(args.scoreboard_source)
    output_path = Path(args.output)

    processed_rows, processed_fields, processed_delim = read_rows(processed_path)
    scoreboard_rows, _, _ = read_rows(scoreboard_path)

    scoreboard_by_id: dict[str, dict[str, str]] = {}
    for row in scoreboard_rows:
        game_id = str(row.get("game_id", "")).strip()
        if not game_id:
            continue
        scoreboard_by_id[game_id] = normalize_scoreboard_row(row)

    output_fields = list(processed_fields)
    for field in SCOREBOARD_FIELDS + DERIVED_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    matched = 0
    for row in processed_rows:
        game_id = str(row.get("game_id", "")).strip()
        payload = scoreboard_by_id.get(game_id, {})
        if payload:
            matched += 1
        for field in SCOREBOARD_FIELDS + DERIVED_FIELDS:
            row[field] = payload.get(field, row.get(field, ""))

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter=processed_delim)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Processed rows: {len(processed_rows)}")
    print(f"Scoreboard rows: {len(scoreboard_by_id)}")
    print(f"Matched by game_id: {matched}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
