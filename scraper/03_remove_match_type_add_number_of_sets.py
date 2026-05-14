#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

INP = "data/tauron_liga_statystyki_merged.csv"   # albo tauron_liga_statystyki.csv
OUT = "tauron_liga_statystyki_final.csv"

DELIM = ";"  # u Ciebie pliki są na średnikach

def main():
    with open(INP, "r", encoding="utf-8-sig", newline="") as f_in:
        r = csv.DictReader(f_in, delimiter=DELIM)
        old_fields = list(r.fieldnames or [])

        # usuwamy match_type
        fieldnames = [c for c in old_fields if c != "match_type"]

        # dodajemy number_of_sets (np. po sety_B jeśli istnieje)
        if "number_of_sets" not in fieldnames:
            if "sety_B" in fieldnames:
                idx = fieldnames.index("sety_B") + 1
                fieldnames.insert(idx, "number_of_sets")
            else:
                fieldnames.append("number_of_sets")

        with open(OUT, "w", encoding="utf-8-sig", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=DELIM)
            w.writeheader()

            for row in r:
                # policz liczbę setów
                try:
                    a = int((row.get("sety_A") or "").strip())
                    b = int((row.get("sety_B") or "").strip())
                    row["number_of_sets"] = a + b
                except:
                    row["number_of_sets"] = ""

                # usuń match_type z wiersza (żeby nie przeszkadzał)
                row.pop("match_type", None)

                # zapisz tylko kolumny docelowe
                out_row = {k: row.get(k, "") for k in fieldnames}
                w.writerow(out_row)

    print("Zapisano:", OUT)

if __name__ == "__main__":
    main()