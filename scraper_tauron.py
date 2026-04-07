#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
 Skrypt do pobierania statystyk TAURON Ligi Kobiet
 Praca licencjacka — zbieranie danych meczowych
 v2 — obsługa PDF + fallback na HTML
=============================================================
 Użycie:
   1. pip install requests beautifulsoup4 pdfplumber
   2. python scraper_tauron.py

 Wynik: tauron_liga_statystyki.csv
=============================================================
"""

import requests, re, time, csv, io, sys, os

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q",
                           "--break-system-packages"])

for pkg, imp in [("requests","requests"),("beautifulsoup4","bs4"),("pdfplumber","pdfplumber")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Instalowanie {pkg}...")
        install(pkg)

import pdfplumber
from bs4 import BeautifulSoup

SEZONY = {
"2015/2016": 24,
    # "2021/2022": 39,
    # "2022/2023": 42,
    # "2023/2024": 45,
    # "2024/2025": 48,
}

BASE_URL   = "https://www.tauronliga.pl"
DELAY      = 1.5
OUTPUT_CSV = "tauron_liga_statystyki.csv"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}

STAT_COLS = [
    "pts_suma","pts_bp","pts_bilans",
    "srv_suma","srv_bledy","srv_asy",
    "rec_suma","rec_bledy","rec_poz_pct","rec_perf_pct",
    "atk_suma","atk_bledy","atk_blok","atk_pkt","atk_skut_pct",
    "blk_pkt",
]

# ══════════════════════════════════════════════════════════════════════════════
# KROK 1 — lista ID meczów
# ══════════════════════════════════════════════════════════════════════════════
def pobierz_id_meczow(tour_id, sezon):
    url = f"{BASE_URL}/games/tour/{tour_id}.html"
    print(f"\n{'='*60}\nSezon {sezon}  (tour {tour_id})\n{'='*60}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ids = sorted(set(re.findall(rf'/games/action/show/id/(\d+)/tour/{tour_id}', resp.text)))
    print(f"  Znaleziono {len(ids)} ID meczów.")
    return ids

# ══════════════════════════════════════════════════════════════════════════════
# KROK 2a — parsowanie PDF
# ══════════════════════════════════════════════════════════════════════════════
def parsuj_pdf(pdf_bytes, game_id, sezon):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return None, f"PDF read error: {e}"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    team1 = team2 = sets1 = sets2 = None
    for line in lines[:8]:
        m = re.match(r'^(.+?)\s+(\d)\s+(\d)\s+(.+)$', line)
        if m:
            team1, sets1, sets2, team2 = m.group(1).strip(), int(m.group(2)), int(m.group(3)), m.group(4).strip()
            break
    if team1 is None:
        return None, "brak wyniku w PDF"

    # EN (nowsze): 'Players total' / PL (starsze): 'Suma zawodnika'
    PT = (r'(?:Players total|Suma zawodnika)\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+'
          r'(\d+)\s+(\d+)\s+(\d+)\s+'
          r'(\d+)\s+(\d+)\s+(\d+)%\s*\(?(\d+)%?\)?\s+'
          r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+(\d+)')
    totals = re.findall(PT, text)
    if len(totals) < 2:
        return None, "brak 2x Players total / Suma zawodnika"

    def to_dict(t):
        return {c: int(v) for c, v in zip(STAT_COLS, t)}

    return _build_rows(game_id, sezon, team1, team2, sets1, sets2,
                       to_dict(totals[0]), to_dict(totals[1]), "PDF"), None

# ══════════════════════════════════════════════════════════════════════════════
# KROK 2b — parsowanie HTML (fallback)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_int(s):
    if s is None:
        return 0
    s = s.strip().replace('%','').replace('+','').replace('\xa0','').replace(' ','')
    if s in ('', '-', '*', '–', '−'):
        return 0
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0

def parsuj_html(html, game_id, sezon):
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text()

    # --- Wykryj mecze bez statystyk (walkover, anulowane) ---
    walkover_markers = ["walkower", "walk over", "w.o.", "walkover",
                        "mecz nie odbył", "brak statystyk", "no stats"]
    if any(m in page_text.lower() for m in walkover_markers):
        return None, "walkower/brak statystyk"

    # Sprawdź czy w ogóle są tabele ze statystykami
    has_stat_tables = any(
        "Suma" in [th.get_text(strip=True) for th in t.find_all("th")]
        and "Błąd" in [th.get_text(strip=True) for th in t.find_all("th")]
        for t in soup.find_all("table")
    )
    if not has_stat_tables:
        # Podaj co faktycznie jest na stronie (pierwsze 200 znaków tekstu)
        snippet = " ".join(page_text.split())[:200]
        return None, f"brak tabel statystyk — strona: {snippet!r}"

    # --- Wynik meczu ---
    sets1 = sets2 = None

    # 1) Szukaj w elemencie z wynikiem (blok wyniku lub cały tekst)
    score_block = soup.find("div", class_=re.compile(r'result|score|wynik', re.I))
    score_text = score_block.get_text() if score_block else page_text

    # Wzorzec: wynik setów 3:x lub x:3 (x in 0,1,2)
    for pat in [r'\b([0-3])\s*:\s*([0-3])\b', r'(\d):(\d)']:
        sm = re.search(pat, score_text)
        if sm:
            a, b = int(sm.group(1)), int(sm.group(2))
            if max(a, b) == 3 and a + b >= 3:
                sets1, sets2 = a, b
                break

    # 2) Fallback — tytuł strony
    if sets1 is None:
        title = soup.find("title")
        if title:
            tm = re.search(r'(\d):(\d)', title.get_text())
            if tm:
                a, b = int(tm.group(1)), int(tm.group(2))
                if max(a, b) == 3:
                    sets1, sets2 = a, b

    # 3) Fallback — przebieg meczu (tabela z "Set" i "Wynik")
    if sets1 is None:
        for tbl in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            if "Set" in headers and "Wynik" in headers:
                rows = tbl.find_all("tr")
                if rows:
                    last = rows[-1].find_all("td")
                    if last:
                        m = re.search(r'(\d)\s*:\s*(\d)', last[-1].get_text())
                        if m:
                            a, b = int(m.group(1)), int(m.group(2))
                            if max(a, b) == 3:
                                sets1, sets2 = a, b
                                break

    if sets1 is None:
        return None, "brak wyniku w HTML (tabele są, ale nie znaleziono wyniku)"

    # nazwy drużyn z nagłówków h3
    h3s = [h.get_text(strip=True) for h in soup.find_all("h3") if h.get_text(strip=True)]
    team_names = [n for n in h3s if len(n) > 3
                  and "Statystyki" not in n and "Legenda" not in n
                  and "Przebieg" not in n and "Szczeg" not in n]

    # ── Dynamiczne wykrywanie kolumn z nagłówków tabeli ──────────────────────
    # Tabela ma 2 rzędy nagłówków:
    #   Rząd 1 (grupy):  Punkty(3) | Zagrywka(4) | Przyjęcie(4) | Atak(6) | Blok(2) | Inne(2)
    #   Rząd 2 (kolumny): Suma | BP | Bilans | Suma | Błąd | As | Eff% | ...
    # "Suma" pojawia się 4 razy → musimy znać kontekst grupy, żeby wiedzieć która to
    # Mapowanie: (słowo_kluczowe_grupy, słowo_kluczowe_kolumny) → nazwa_statu

    # ── Indeksy STAT liczone od prawej (niezależne od formatu nr/imię) ─────────
    # Ostatnie 21 td w każdym wierszu zawodniczki to zawsze te same kolumny:
    # pos od końca: -21=pts_suma, -20=pts_bp, -19=pts_bilans,
    #               -18=srv_suma, -17=srv_bledy, -16=srv_asy, -15=srv_eff%,
    #               -14=rec_suma, -13=rec_bledy, -12=rec_poz%, -11=rec_perf%,
    #               -10=atk_suma, -9=atk_bledy, -8=atk_blok, -7=atk_pkt,
    #               -6=atk_skut%, -5=atk_eff%, -4=blk_pkt, -3=blk_wyblok,
    #               -2=obrona, -1=asysta
    STAT_RIDX = {           # indeks od prawej (ujemny)
        "pts_suma":  -21,
        "pts_bp":    -20,
        "pts_bilans":-19,
        "srv_suma":  -18,
        "srv_bledy": -17,
        "srv_asy":   -16,
        "rec_suma":  -14,
        "rec_bledy": -13,
        # rec_poz_pct i rec_perf_pct — czytamy z wiersza sumy (indeksy -12, -11)
        "rec_poz_pct":  -12,
        "rec_perf_pct": -11,
        "atk_suma":  -10,
        "atk_bledy":  -9,
        "atk_blok":   -8,
        "atk_pkt":    -7,
        # atk_skut_pct — przeliczamy z atk_pkt/atk_suma
        "blk_pkt":    -4,
    }
    N_STAT_COLS = 21   # zawsze 21 kolumn statystyk w tabeli tauronligi
    # Klucze których NIE sumujemy po zawodniczkach — bierzemy z wiersza sumy
    PCT_FROM_TOTAL = {"rec_poz_pct", "rec_perf_pct"}

    def _has_stat_headers(tbl):
        """Czy tabela ma nagłówki statystyczne (Suma + Błąd)?"""
        th_texts = {th.get_text(strip=True) for th in tbl.find_all("th")}
        return "Suma" in th_texts and "Błąd" in th_texts

    def _sum_table(tbl):
        """Zsumuj statystyki zawodniczek; procenty przyjęcia z wiersza sumy."""
        if not _has_stat_headers(tbl):
            return None

        totals = {c: 0 for c in STAT_COLS}
        player_count = 0
        SKIP_KW = ["Suma z", "Punkty", "Zagrywka", "total", "zawodnika",
                   "Players total", "Suma zawodnika"]

        # Liczniki do ważonej średniej procentów przyjęcia
        rec_pos_sum  = 0.0
        rec_perf_sum = 0.0

        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            vals = [td.get_text(strip=True) for td in tds]
            first = vals[0] if vals else ""

            # Pomiń wiersze nagłówkowe i sumy
            if any(kw.lower() in first.lower() for kw in SKIP_KW):
                continue

            if len(vals) < N_STAT_COLS + 1:
                continue

            def rv(ridx):
                return _safe_int(vals[ridx]) if len(vals) >= abs(ridx) else 0

            # Zawodniczka grała jeśli ma jakiekolwiek niezerowe statystyki
            played = (rv(-21) != 0 or rv(-18) > 0 or rv(-10) > 0
                      or rv(-19) != 0 or rv(-14) > 0)
            if not played:
                continue

            # Sumuj kolumny liczbowe
            for stat, ridx in STAT_RIDX.items():
                if stat in PCT_FROM_TOTAL:
                    continue
                if len(vals) >= abs(ridx):
                    totals[stat] += _safe_int(vals[ridx])

            # Zbierz dane do średniej ważonej procentów przyjęcia
            rec = rv(-14)
            if rec > 0:
                poz_str  = vals[-12] if len(vals) >= 12 else "0"
                perf_str = vals[-11] if len(vals) >= 11 else "0"
                rec_pos_sum  += _safe_int(poz_str)  * rec
                rec_perf_sum += _safe_int(perf_str) * rec

            player_count += 1

        # Oblicz procenty przyjęcia jako średnią ważoną (waga = rec_suma)
        if totals["rec_suma"] > 0:
            totals["rec_poz_pct"]  = round(rec_pos_sum  / totals["rec_suma"])
            totals["rec_perf_pct"] = round(rec_perf_sum / totals["rec_suma"])

        return totals if player_count > 0 else None

    # Znajdź tabele statystyk i zsumuj
    stat_tables_parsed = []
    for t in soup.find_all("table"):
        ts = _sum_table(t)
        if ts is not None:
            stat_tables_parsed.append(ts)

    if len(stat_tables_parsed) < 2:
        snippet = " ".join(page_text.split())[:200]
        return None, f"znaleziono tylko {len(stat_tables_parsed)} tabel statystyk — strona: {snippet!r}"

    team_stats = []
    for ts in stat_tables_parsed[:2]:
        # Przelicz atk_skut_pct z surowych liczb
        ts["atk_skut_pct"] = (round(ts["atk_pkt"] / ts["atk_suma"] * 100)
                               if ts["atk_suma"] > 0 else 0)
        team_stats.append(ts)

    if len(team_stats) < 2:
        return None, "nie udało się zebrać statystyk dla 2 drużyn"

    t1 = team_names[0] if len(team_names) > 0 else "Druzyna1"
    t2 = team_names[1] if len(team_names) > 1 else "Druzyna2"

    return _build_rows(game_id, sezon, t1, t2, sets1, sets2,
                       team_stats[0], team_stats[1], "HTML"), None

# ══════════════════════════════════════════════════════════════════════════════
# Budowanie wierszy CSV
# ══════════════════════════════════════════════════════════════════════════════
def _build_rows(game_id, sezon, team1, team2, sets1, sets2, s1, s2, source):
    match_type = f"3:{min(sets1,sets2)}"
    row = {
        "game_id": game_id, "sezon": sezon, "source": source,
        "match_type": match_type,
        "druzyna_A": team1, "druzyna_B": team2,
        "sety_A": sets1, "sety_B": sets2,
        "wygrana_A": 1 if sets1 > sets2 else 0,
    }
    for k, v in s1.items():
        row[f"A_{k}"] = v
    for k, v in s2.items():
        row[f"B_{k}"] = v
    for k in s1:
        row[f"diff_{k}"] = s1[k] - s2[k]
    return [row]

# ══════════════════════════════════════════════════════════════════════════════
# KROK 3 — PDF → HTML fallback
# ══════════════════════════════════════════════════════════════════════════════
def pobierz_mecz(game_id, sezon, tour_id):
    # próba PDF
    pdf_err = "nie próbowano"
    try:
        r = requests.get(f"{BASE_URL}/games/action/stats/id/{game_id}/tour/{tour_id}.html",
                         headers=HEADERS, timeout=30)
        ct = r.headers.get("Content-Type","")
        if r.status_code == 200 and ("pdf" in ct.lower() or r.content[:4] == b"%PDF"):
            rows, err = parsuj_pdf(r.content, game_id, sezon)
            if rows:
                return rows, "PDF", None
            pdf_err = err or "parsowanie PDF nie powiodlo sie"
        else:
            pdf_err = f"brak PDF (status={r.status_code}, ct={ct[:40]})"
    except Exception as e:
        pdf_err = str(e)

    # fallback HTML
    try:
        r = requests.get(f"{BASE_URL}/games/action/show/id/{game_id}/tour/{tour_id}.html",
                         headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None, None, f"HTML HTTP {r.status_code}"
        rows, err = parsuj_html(r.text, game_id, sezon)
        if rows:
            return rows, "HTML", None
        # Zapisz zrzut HTML dla pierwszych 20 nieudanych meczów (diagnostyka)
        dump_dir = "html_dumps"
        existing = len(os.listdir(dump_dir)) if os.path.isdir(dump_dir) else 0
        if existing < 20:
            os.makedirs(dump_dir, exist_ok=True)
            with open(f"{dump_dir}/{game_id}.html", "w", encoding="utf-8") as f:
                f.write(r.text)
        return None, None, f"PDF: {pdf_err} | HTML: {err}"
    except Exception as e:
        return None, None, f"PDF: {pdf_err} | HTML exception: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# GŁÓWNA PĘTLA
# ══════════════════════════════════════════════════════════════════════════════
def main():
    wszystkie, bledy = [], []

    for sezon, tour_id in SEZONY.items():
        ids = pobierz_id_meczow(tour_id, sezon)
        time.sleep(DELAY)

        for i, gid in enumerate(ids, 1):
            print(f"  [{sezon}] {i:3d}/{len(ids)} ID={gid} ...", end=" ", flush=True)
            rows, source, err = pobierz_mecz(gid, sezon, tour_id)

            if rows:
                t = rows[0]
                print(f"[{source}] {t['druzyna_A'][:18]} {t['sety_A']}:{t['sety_B']} {t['druzyna_B'][:18]}")
                wszystkie.extend(rows)
            else:
                print(f"SKIP — {err}")
                bledy.append({"sezon": sezon, "game_id": gid, "blad": err})

            time.sleep(DELAY)

    if not wszystkie:
        print("\n[!] Brak danych."); return

    fieldnames = list(wszystkie[0].keys())

    # Wczytaj istniejące game_id żeby nie dublować
    istniejace_ids = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                istniejace_ids.add(row.get("game_id", ""))
        print(f"  Plik istnieje — {len(istniejace_ids)} meczów już w bazie, pomijam duplikaty.")

    nowe = [r for r in wszystkie if str(r["game_id"]) not in istniejace_ids]
    print(f"  Nowych meczów do dopisania: {len(nowe)}")

    if nowe:
        plik_istnieje = os.path.exists(OUTPUT_CSV)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not plik_istnieje:
                w.writeheader()
            w.writerows(nowe)

    if bledy:
        bledy_istnieja = os.path.exists("tauron_bledy.csv")
        with open("tauron_bledy.csv", "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["sezon","game_id","blad"])
            if not bledy_istnieja:
                w.writeheader()
            w.writerows(bledy)

    pdf_n  = sum(1 for r in wszystkie if r.get("source")=="PDF")
    html_n = sum(1 for r in wszystkie if r.get("source")=="HTML")
    print(f"\n{'='*60}")
    print(f"GOTOWE: {len(wszystkie)} meczów ({pdf_n} PDF + {html_n} HTML) → {OUTPUT_CSV}")
    if bledy:
        print(f"Pominięte: {len(bledy)} → tauron_bledy.csv")
    print("="*60)

if __name__ == "__main__":
    main()