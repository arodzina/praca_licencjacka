#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_pobierz_daty_meczow.py

Scrapes match dates from the TAURON Liga website for all seasons
and saves a mapping of game_id → date to a CSV file.

Usage:
    python scraper/04_pobierz_daty_meczow.py
    # Output: data/match_dates.csv
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.tauronliga.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}
DELAY = 0.3

SEASON_TOUR = {
    "2015/2016": 24,
    "2016/2017": 26,
    "2017/2018": 28,
    "2018/2019": 30,
    "2019/2020": 33,
    "2020/2021": 36,
    "2021/2022": 39,
    "2022/2023": 42,
    "2023/2024": 45,
    "2024/2025": 48,
    "2025/2026": 50,
}

OUTPUT_CSV = "data/match_dates.csv"


def scrape_season(season: str, tour: int) -> list[dict]:
    """Fetch all matches for a given season/tour with their game_id and date."""
    url = f"{BASE_URL}/games/tour/{tour}.html"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    matches = []
    for box in soup.find_all("div", class_="game-box"):
        link = box.find("a", href=re.compile(r"/games/action/show/id/\d+"))
        if not link:
            continue
        href = link.get("href", "")
        gid_match = re.search(r"/games/action/show/id/(\d+)", href)
        if not gid_match:
            continue
        gid = int(gid_match.group(1))

        date_el = box.find(class_="game-date")
        if not date_el:
            continue
        date_str = date_el.text.strip().split(",")[0].strip()

        matches.append({"game_id": gid, "season": season, "date": date_str})

    return matches


def main():
    all_matches = []

    for season, tour in sorted(SEASON_TOUR.items()):
        try:
            matches = scrape_season(season, tour)
            all_matches.extend(matches)
        except Exception as e:
            print(f"Error scraping {season}: {e}")
        time.sleep(DELAY)

    all_matches.sort(key=lambda x: x["game_id"])

    fieldnames = ["game_id", "season", "date"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        w.writeheader()
        w.writerows(all_matches)

    print(f"Done. Saved {len(all_matches)} dates to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
