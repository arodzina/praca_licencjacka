#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_usun_typ_meczu_dodaj_liczbe_setow.py

Removes the 'match_type' column and adds 'number_of_sets'
(computed as sety_A + sety_B) to a CSV file.

Usage:
    python scraper/03_usun_typ_meczu_dodaj_liczbe_setow.py
"""

import csv

INP = "data/tauron_liga_statystyki_merged.csv"
OUT = "tauron_liga_statystyki_final.csv"
DELIM = ";"


def main():
    with open(INP, "r", encoding="utf-8-sig", newline="") as f_in:
        r = csv.DictReader(f_in, delimiter=DELIM)
        old_fields = list(r.fieldnames or [])

        fieldnames = [c for c in old_fields if c != "match_type"]

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
                try:
                    a = int((row.get("sety_A") or "").strip())
                    b = int((row.get("sety_B") or "").strip())
                    row["number_of_sets"] = a + b
                except Exception:
                    row["number_of_sets"] = ""

                row.pop("match_type", None)
                out_row = {k: row.get(k, "") for k in fieldnames}
                w.writerow(out_row)

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
