#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

OLD = "data/tauron_liga_statystyki.csv"        # separator ;
UPD = "scoreboard_update.csv"             # separator ;
OUT = "tauron_liga_statystyki_merged.csv" # separator ;

SCORE_FIELDS = [
    "A_set1_points","B_set1_points",
    "A_set2_points","B_set2_points",
    "A_set3_points","B_set3_points",
    "A_set4_points","B_set4_points",
    "A_set5_points","B_set5_points",
    "A_scoreboard_points","B_scoreboard_points",
]

def main():
    if not os.path.exists(OLD):
        raise FileNotFoundError(f"Brak pliku OLD: {OLD}")
    if not os.path.exists(UPD):
        raise FileNotFoundError(f"Brak pliku UPD: {UPD}")

    # 1) Wczytaj update (po game_id)
    upd = {}
    with open(UPD, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")  # <-- UPD też ma średniki
        if not r.fieldnames or "game_id" not in r.fieldnames:
            raise ValueError(f"UPD nie ma kolumny game_id albo ma zły delimiter. Fieldnames={r.fieldnames}")

        for row in r:
            gid = (row.get("game_id") or "").strip()
            if gid:
                upd[gid] = row

    # 2) Przejdź po starym pliku i dopisz/uzupełnij kolumny
    with open(OLD, "r", encoding="utf-8-sig", newline="") as f_in:
        r_old = csv.DictReader(f_in, delimiter=";")
        if not r_old.fieldnames or "game_id" not in r_old.fieldnames:
            raise ValueError(f"OLD nie ma kolumny game_id albo ma zły delimiter. Fieldnames={r_old.fieldnames}")

        fieldnames = list(r_old.fieldnames)

        for c in SCORE_FIELDS:
            if c not in fieldnames:
                fieldnames.append(c)

        with open(OUT, "w", newline="", encoding="utf-8-sig") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=";")
            w.writeheader()

            merged_cnt = 0
            for row in r_old:
                gid = (row.get("game_id") or "").strip()
                u = upd.get(gid)

                if u:
                    for c in SCORE_FIELDS:
                        val = (u.get(c) or "").strip()
                        if val != "":
                            row[c] = val
                    merged_cnt += 1

                w.writerow(row)

    print("Zapisano:", OUT)
    print("Zmerge'owanych game_id (znalezionych w UPD):", merged_cnt)

if __name__ == "__main__":
    main()