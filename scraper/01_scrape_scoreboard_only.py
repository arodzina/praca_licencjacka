#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_scrape_scoreboard_only.py

- Czyta game_id ze starego pliku (separator ;), np. tauron_liga_statystyki.csv
- Scrape’uje scoreboard (punkty w setach + suma) ze strony meczu
- Dopisuje do istniejącego pliku wynikowego (separator ,) i robi RESUME:
  nie pobiera ponownie ID, które już są w OUT_CSV.

Uwaga:
- OUT_CSV zapisuje z PRZECINKAMI (,)
- IN_CSV może być ze średnikami (;), bo taki masz.
"""

import csv
import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.tauronliga.pl"
HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}
DELAY    = 1.0

IN_CSV   = "data/tauron_liga_statystyki.csv"   # stary plik (u Ciebie ; )
OUT_CSV  = "scoreboard_update.csv"        # nowy plik (z przecinkami)

SCORE_FIELDS = [
    "game_id",
    "A_set1_points","B_set1_points",
    "A_set2_points","B_set2_points",
    "A_set3_points","B_set3_points",
    "A_set4_points","B_set4_points",
    "A_set5_points","B_set5_points",
    "A_scoreboard_points","B_scoreboard_points",
]

def extract_set_scores_from_html(soup: BeautifulSoup):
    """Zwraca listę par [(A,B), ...] z tabeli przebiegu (kolumna Punkty)."""
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
    out = {k: "" for k in SCORE_FIELDS if k != "game_id"}
    for i in range(1, 6):
        out[f"A_set{i}_points"] = ""
        out[f"B_set{i}_points"] = ""
    out["A_scoreboard_points"] = ""
    out["B_scoreboard_points"] = ""

    if not pairs:
        return out

    for i, (a, b) in enumerate(pairs[:5], start=1):
        out[f"A_set{i}_points"] = a
        out[f"B_set{i}_points"] = b
    out["A_scoreboard_points"] = sum(a for a, _ in pairs)
    out["B_scoreboard_points"] = sum(b for _, b in pairs)
    return out

def load_game_ids_from_old_csv(path: str):
    ids = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")  # <-- u Ciebie stary plik jest na ;
        for row in r:
            gid = (row.get("game_id") or "").strip()
            if gid:
                ids.append(gid)
    # unikat + sort
    ids = sorted(set(ids), key=lambda x: int(x))
    return ids

def load_done_ids_from_out_csv(path: str):
    """OUT ma przecinki, więc delimiter=','."""
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
    ids = load_game_ids_from_old_csv(IN_CSV)
    print("ID do zrobienia (w starym pliku):", len(ids))

    done = load_done_ids_from_out_csv(OUT_CSV)
    if done:
        print("Już zapisane w OUT_CSV (resume):", len(done))

    file_exists = os.path.exists(OUT_CSV)

    # dopisujemy do pliku (append)
    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f_out:
        w = csv.DictWriter(f_out, fieldnames=SCORE_FIELDS, delimiter=";")
        if not file_exists:
            w.writeheader()

        left = [gid for gid in ids if gid not in done]
        print("Pozostało do zrobienia:", len(left))

        for i, gid in enumerate(left, 1):
            url = f"{BASE_URL}/games/action/show/id/{gid}.html"  # bez tour też działa
            row = {"game_id": gid}

            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code != 200:
                    print(f"{i}/{len(left)} ID={gid} HTTP {resp.status_code}")
                    w.writerow(row)
                    time.sleep(DELAY)
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                pairs = extract_set_scores_from_html(soup)
                row.update(scores_to_fields(pairs))
                w.writerow(row)

                if row.get("A_scoreboard_points") != "":
                    print(f"{i}/{len(left)} ID={gid} SB {row['A_scoreboard_points']}:{row['B_scoreboard_points']}")
                else:
                    print(f"{i}/{len(left)} ID={gid} brak scoreboard")

            except Exception as e:
                print(f"{i}/{len(left)} ID={gid} EXC {e}")
                w.writerow(row)

            time.sleep(DELAY)

    print("Gotowe. Wynik w:", OUT_CSV)

if __name__ == "__main__":
    main()