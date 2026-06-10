#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_pobierz_wyniki_setow.py

Scrapes per-set points and total scoreboard points for matches
listed in an existing CSV file and appends them to an output CSV.

Usage:
    python scraper/01_pobierz_wyniki_setow.py
"""

import csv
import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.tauronliga.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}
DELAY = 1.0

IN_CSV = "data/tauron_liga_statystyki.csv"
OUT_CSV = "scoreboard_update.csv"

SCORE_FIELDS = [
    "game_id",
    "A_set1_points", "B_set1_points",
    "A_set2_points", "B_set2_points",
    "A_set3_points", "B_set3_points",
    "A_set4_points", "B_set4_points",
    "A_set5_points", "B_set5_points",
    "A_scoreboard_points", "B_scoreboard_points",
]


def extract_set_scores_from_html(soup: BeautifulSoup):
    """Extract per-set point pairs [(A, B), ...] from the match progress table."""
    def parse_pair(txt: str):
        m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", txt)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a < 10 and b < 10:
            return None
        return (a, b)

    for tbl in soup.find_all("table"):
        ths = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        if not ths:
            continue
        if "set" not in ths:
            continue

        punkty_idx = None
        for i, h in enumerate(ths):
            if "punkty" in h:
                punkty_idx = i
                break
        if punkty_idx is None:
            continue

        pairs = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) <= punkty_idx:
                continue
            cell = tds[punkty_idx].get_text(" ", strip=True)
            pair = parse_pair(cell)
            if pair:
                pairs.append(pair)

        if pairs:
            return pairs
    return []


def scores_to_fields(pairs):
    """Convert a list of set score pairs into a flat dictionary of SCORE_FIELDS."""
    out = {k: "" for k in SCORE_FIELDS if k != "game_id"}
    if not pairs:
        return out

    for i, (a, b) in enumerate(pairs[:5], start=1):
        out[f"A_set{i}_points"] = a
        out[f"B_set{i}_points"] = b
    out["A_scoreboard_points"] = sum(a for a, _ in pairs)
    out["B_scoreboard_points"] = sum(b for _, b in pairs)
    return out


def load_game_ids(path: str):
    """Load all game_id values from a CSV file."""
    ids = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            gid = (row.get("game_id") or "").strip()
            if gid:
                ids.append(gid)
    return sorted(set(ids), key=lambda x: int(x))


def load_done_ids(path: str):
    """Load game_id values already present in the output file (for resume)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            gid = (row.get("game_id") or "").strip()
            if gid:
                done.add(gid)
    return done


def main():
    ids = load_game_ids(IN_CSV)
    done = load_done_ids(OUT_CSV)
    remaining = [gid for gid in ids if gid not in done]

    file_exists = os.path.exists(OUT_CSV)

    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f_out:
        w = csv.DictWriter(f_out, fieldnames=SCORE_FIELDS, delimiter=";")
        if not file_exists:
            w.writeheader()

        for i, gid in enumerate(remaining, 1):
            url = f"{BASE_URL}/games/action/show/id/{gid}.html"
            row = {"game_id": gid}

            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code != 200:
                    w.writerow(row)
                    time.sleep(DELAY)
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                pairs = extract_set_scores_from_html(soup)
                row.update(scores_to_fields(pairs))
                w.writerow(row)

            except Exception:
                w.writerow(row)

            time.sleep(DELAY)

    print(f"Done. {len(remaining)} matches processed → {OUT_CSV}")


if __name__ == "__main__":
    main()
