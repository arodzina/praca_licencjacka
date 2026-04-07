import requests, csv, os

# Twój istniejący scraper — zaimportuj z niego potrzebne funkcje
from scraper_tauron import pobierz_mecz, OUTPUT_CSV

game_id = "6686"
sezon = "2015/2016"   # ← zmień jeśli inny sezon
tour_id = 24           # ← zmień jeśli inny tour

rows, source, err = pobierz_mecz(game_id, sezon, tour_id)

if rows:
    row = rows[0]
    print(f"[{source}] {row['druzyna_A']} {row['sety_A']}:{row['sety_B']} {row['druzyna_B']}")

    # Dopisz do CSV
    plik_istnieje = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not plik_istnieje:
            w.writeheader()
        w.writerow(row)
    print(f"Dopisano do {OUTPUT_CSV}")
else:
    print(f"Błąd: {err}")