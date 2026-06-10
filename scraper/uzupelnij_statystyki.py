#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
Narzędzie do ręcznego uzupełniania statystyk meczów 2025/2026
================================================================

Sposób użycia:
  1. Znajdź game_id meczu (z listy poniżej lub z CSV)
  2. Uruchom: python scraper/uzupelnij_statystyki.py --id GAME_ID
  3. Wklej statystyki w formacie CSV (pytanie po pytaniu)

ALBO wklej wszystkie naraz:
  python scraper/uzupelnij_statystyki.py --id GAME_ID --stats "A_pts_suma=60,A_pts_bp=25,..."

Lista meczów bez statystyk:
"""

import csv, sys, os, re

CSV_FILE = "data/tauron_liga_statystyki_2025_2026.csv"

# Statystyki do uzupełnienia (A i B)
STAT_KEYS = [
    "pts_suma", "pts_bp", "pts_bilans",
    "srv_suma", "srv_bledy", "srv_asy",
    "rec_suma", "rec_bledy", "rec_poz_pct", "rec_perf_pct",
    "atk_suma", "atk_bledy", "atk_blok", "atk_pkt", "atk_skut_pct",
    "blk_pkt",
]

def show_missing():
    """Wyświetl listę meczów bez statystyk."""
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        missing = [r for r in reader if r.get("source", "").strip() == "HTML_SCOREBOARD_ONLY"]
    
    if not missing:
        print(" Wszystkie mecze mają statystyki!")
        return
    
    print(f"\nMecze bez statystyk ({len(missing)}):")
    print("-" * 80)
    print(f"{'ID':>12} | {'Drużyna A':25s} | {'Drużyna B':25s} | Wynik")
    print("-" * 80)
    for r in missing:
        print(f"{r['game_id']:>12} | {r['druzyna_A'][:25]:25s} | {r['druzyna_B'][:25]:25s} | {r['sety_A']}:{r['sety_B']}")
    print("-" * 80)

def update_stats(game_id, stats_dict):
    """Zaktualizuj statystyki dla danego game_id."""
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    found = False
    for row in rows:
        if row["game_id"] == str(game_id):
            for k, v in stats_dict.items():
                if k in row:
                    row[k] = v
            row["source"] = "MANUAL"
            found = True
            break
    
    if not found:
        print(f"❌ Nie znaleziono meczu o ID={game_id}")
        return
    
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        w.writeheader()
        w.writerows(rows)
    
    print(f"✅ Zaktualizowano mecz ID={game_id}")

def interactive_update(game_id):
    """Interaktywne uzupełnianie statystyk."""
    print(f"\nUzupełnianie statystyk dla meczu ID={game_id}")
    print("=" * 50)
    
    stats = {}
    
    for prefix in ["A", "B"]:
        print(f"\n--- Statystyki drużyny {prefix} ---")
        for key in STAT_KEYS:
            col = f"{prefix}_{key}"
            val = input(f"  {col}: ").strip()
            if val:
                stats[col] = val
    
    # Oblicz diff
    for key in STAT_KEYS:
        a_key = f"A_{key}"
        b_key = f"B_{key}"
        diff_key = f"diff_{key}"
        
        a_val = stats.get(a_key, "")
        b_val = stats.get(b_key, "")
        
        if a_val and b_val:
            try:
                a_num = float(a_val) if '.' in a_val else int(a_val)
                b_num = float(b_val) if '.' in b_val else int(b_val)
                stats[diff_key] = a_num - b_num
            except ValueError:
                pass
    
    print(f"\nStatystyki do zapisania:")
    for k, v in stats.items():
        print(f"  {k} = {v}")
    
    confirm = input("\nZapisać? (t/N): ").strip().lower()
    if confirm == 't':
        update_stats(game_id, stats)
    else:
        print("Anulowano.")

def quick_update(game_id, stats_csv):
    """Szybka aktualizacja z ciągu 'key=val,key=val,...'."""
    stats = {}
    for pair in stats_csv.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            stats[k.strip()] = v.strip()
    
    if not stats:
        print(" Nie podano statystyk")
        return
    
    update_stats(game_id, stats)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        show_missing()
        sys.exit(0)
    
    if sys.argv[1] == "--list":
        show_missing()
    elif sys.argv[1] == "--id" and len(sys.argv) >= 3:
        game_id = sys.argv[2]
        if "--stats" in sys.argv:
            idx = sys.argv.index("--stats")
            stats_str = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
            quick_update(game_id, stats_str)
        else:
            interactive_update(game_id)
    else:
        print("Użycie: python scraper/uzupelnij_statystyki.py --id GAME_ID")
        print("  lub:  python scraper/uzupelnij_statystyki.py --list")
