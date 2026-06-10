#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_scal_wyniki_setow.py

Merges scoreboard columns (per-set points) from a scoreboard CSV file
into the main statistics CSV by matching game_id.

Usage:
    python scraper/02_scal_wyniki_setow.py
"""

import csv
import os

OLD = "data/tauron_liga_statystyki.csv"
UPD = "scoreboard_update.csv"
OUT = "tauron_liga_statystyki_merged.csv"

SCORE_FIELDS = [
    "A_set1_points", "B_set1_points",
    "A_set2_points", "B_set2_points",
    "A_set3_points", "B_set3_points",
    "A_set4_points", "B_set4_points",
    "A_set5_points", "B_set5_points",
    "A_scoreboard_points", "B_scoreboard_points",
]


def main():
    if not os.path.exists(OLD):
        raise FileNotFoundError(f"Missing file: {OLD}")
    if not os.path.exists(UPD):
        raise FileNotFoundError(f"Missing file: {UPD}")

    # Load scoreboard updates indexed by game_id
    updates = {}
    with open(UPD, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        if not r.fieldnames or "game_id" not in r.fieldnames:
            raise ValueError(f"UPD missing game_id column. Fields: {r.fieldnames}")
        for row in r:
            gid = (row.get("game_id") or "").strip()
            if gid:
                updates[gid] = row

    # Read old file and merge
    with open(OLD, "r", encoding="utf-8-sig", newline="") as f_in:
        r_old = csv.DictReader(f_in, delimiter=";")
        if not r_old.fieldnames or "game_id" not in r_old.fieldnames:
            raise ValueError(f"OLD missing game_id column. Fields: {r_old.fieldnames}")

        fieldnames = list(r_old.fieldnames)
        for c in SCORE_FIELDS:
            if c not in fieldnames:
                fieldnames.append(c)

        merged_count = 0
        with open(OUT, "w", newline="", encoding="utf-8-sig") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=";")
            w.writeheader()

            for row in r_old:
                gid = (row.get("game_id") or "").strip()
                u = updates.get(gid)
                if u:
                    for c in SCORE_FIELDS:
                        val = (u.get(c) or "").strip()
                        if val != "":
                            row[c] = val
                    merged_count += 1
                w.writerow(row)

    print(f"Saved: {OUT}")
    print(f"Merged game_id values (found in UPD): {merged_count}")


if __name__ == "__main__":
    main()
