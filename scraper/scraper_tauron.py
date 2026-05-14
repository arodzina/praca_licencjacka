#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
Skrypt do pobierania statystyk TAURON Ligi Kobiet
Praca licencjacka — zbieranie danych meczowych

v2 — obsługa PDF + fallback na HTML

UPDATE (scoreboard + sety):
- punkty w setach + sumy scoreboard:
  * A_set1_points ... A_set5_points
  * B_set1_points ... B_set5_points
  * A_scoreboard_points, B_scoreboard_points
- zamiast match_type:
  * sets_played = sety_A + sety_B

UPDATE (stabilny CSV):
- stała kolejność kolumn: FIELDNAMES
- dopełnianie brakujących pól: row.setdefault(...)

UPDATE (fallback):
- PDF jest źródłem nr 1
- gdy PDF nie ma/nie da się sparsować:
  1) próbuj pełnych statystyk z HTML (parsuj_html)
  2) jeśli brak tabel w HTML -> scoreboard-only (parsuj_html_scoreboard_only)

Użycie:
  1. pip install requests beautifulsoup4 pdfplumber
  2. python scraper_tauron.py
============================================================
"""

import requests, re, time, csv, io, sys, os

def install(pkg):
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", pkg, "-q", "--break-system-packages"
    ])

for pkg, imp in [("requests", "requests"), ("beautifulsoup4", "bs4"), ("pdfplumber", "pdfplumber")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Instalowanie {pkg}...")
        install(pkg)

import pdfplumber
from bs4 import BeautifulSoup

SEZONY = {
    "2025/2026": 50,
}

BASE_URL   = "https://www.tauronliga.pl"
DELAY      = 1.0
OUTPUT_CSV = "data/tauron_liga_statystyki_2025_2026.csv"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}

STAT_COLS = [
    "pts_suma","pts_bp","pts_bilans",
    "srv_suma","srv_bledy","srv_asy",
    "rec_suma","rec_bledy","rec_poz_pct","rec_perf_pct",
    "atk_suma","atk_bledy","atk_blok","atk_pkt","atk_skut_pct",
    "blk_pkt",
]

# ─────────────────────────────────────────────────────────────────────────────
# Stała kolejność kolumn w CSV (żeby nic się nie przesuwało)
# ─────────────────────────────────────────────────────────────────────────────
BASE_FIELDS = [
    "game_id","sezon","source",
    "druzyna_A","druzyna_B",
    "sety_A","sety_B","sets_played","wygrana_A",
    "A_set1_points","B_set1_points",
    "A_set2_points","B_set2_points",
    "A_set3_points","B_set3_points",
    "A_set4_points","B_set4_points",
    "A_set5_points","B_set5_points",
    "A_scoreboard_points","B_scoreboard_points",
]

STAT_FIELDS = []
for side in ["A", "B"]:
    for k in STAT_COLS:
        STAT_FIELDS.append(f"{side}_{k}")
for k in STAT_COLS:
    STAT_FIELDS.append(f"diff_{k}")

FIELDNAMES = BASE_FIELDS + STAT_FIELDS

# ─────────────────────────────────────────────────────────────────────────────
# Scoreboard helpers (punkty w setach)
# ─────────────────────────────────────────────────────────────────────────────
def _set_scores_to_row_fields(set_scores):
    out = {}
    for i in range(1, 6):
        out[f"A_set{i}_points"] = ""
        out[f"B_set{i}_points"] = ""
    out["A_scoreboard_points"] = ""
    out["B_scoreboard_points"] = ""
    if not set_scores:
        return out

    for i, (a, b) in enumerate(set_scores[:5], start=1):
        out[f"A_set{i}_points"] = a
        out[f"B_set{i}_points"] = b
    out["A_scoreboard_points"] = sum(a for a, _ in set_scores)
    out["B_scoreboard_points"] = sum(b for _, b in set_scores)
    return out

def _extract_set_scores_from_html(soup: BeautifulSoup):
    """
    Tabela 'Przebieg meczu':
    - 'Punkty' = wynik punktowy seta (np. 25 : 22) ✅
    - 'Wynik'  = stan setów (np. 2 : 0)           ❌
    """
    def parse_pair(text: str):
        m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
        if not m:
            return None
        a = int(m.group(1)); b = int(m.group(2))
        if a < 10 and b < 10:
            return None
        return (a, b)

    for tbl in soup.find_all("table"):
        ths = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        if not ths:
            continue
        norm = [t.replace("\xa0", " ").strip() for t in ths]
        if not any(t == "set" for t in norm):
            continue
        if not any("punkty" in t for t in norm):
            continue

        punkty_idx = None
        for i, t in enumerate(norm):
            if "punkty" in t:
                punkty_idx = i
                break
        if punkty_idx is None:
            continue

        pairs = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) <= punkty_idx:
                continue
            cell_text = tds[punkty_idx].get_text(" ", strip=True)
            pair = parse_pair(cell_text)
            if pair:
                pairs.append(pair)

        if pairs:
            return pairs

    return []

def _find_main_game_node(soup: BeautifulSoup, game_id):
    """
    Na stronie meczu bywa dużo innych boxów z listy spotkań.
    Szukamy kontenera aktualnego meczu po data-game-id, żeby nie mieszać scoreboardów.
    """
    selectors = [
        f'.ajax-synced-games[data-game-id="{game_id}"]',
        f'[data-game-id="{game_id}"]',
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return soup

def _extract_set_scores_from_scoreboard_box(soup: BeautifulSoup, game_id):
    """
    Fallback dla nowszych stron: punkty setowe są w boxie meczu jako span-y
    data-synced-games-content="set1pointsTeamA" itd., nawet gdy brak tabeli przebiegu.
    """
    root = _find_main_game_node(soup, game_id)
    pairs = []

    for set_no in range(1, 6):
        a_node = root.select_one(
            f'[data-synced-games-content="set{set_no}pointsTeamA"]'
        )
        b_node = root.select_one(
            f'[data-synced-games-content="set{set_no}pointsTeamB"]'
        )
        if not a_node or not b_node:
            continue

        a_txt = a_node.get_text(" ", strip=True)
        b_txt = b_node.get_text(" ", strip=True)
        if not a_txt.isdigit() or not b_txt.isdigit():
            continue

        a = int(a_txt)
        b = int(b_txt)

        # Puste niegrane sety bywają renderowane jako 0:0.
        if a == 0 and b == 0:
            continue

        pairs.append((a, b))

    return pairs

def _extract_teams_from_match_box(soup: BeautifulSoup, game_id):
    """
    Dla stron z widgetami volleystation nazwy drużyn są w subtitle i game-team.
    To jest stabilniejsze niż parsowanie <title>.
    """
    root = _find_main_game_node(soup, game_id)

    subtitle = soup.select_one(f'.subtitle.ajax-synced-games[data-game-id="{game_id}"]')
    if subtitle:
        vs = subtitle.find("small", class_="vs")
        if vs:
            team1 = vs.previous_sibling.strip() if isinstance(vs.previous_sibling, str) else ""
            team2 = vs.next_sibling.strip() if isinstance(vs.next_sibling, str) else ""
            team1 = re.sub(r"\s+", " ", team1).strip()
            team2 = re.sub(r"\s+", " ", team2).strip()
            if team1 and team2:
                return team1, team2

    left = root.select_one(".game-team.left")
    right = root.select_one(".game-team.right")
    if left and right:
        return left.get_text(" ", strip=True), right.get_text(" ", strip=True)

    title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
    m = re.search(r"^(.*?)\s+vs\s+(.*?)\s+-", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return "DruzynaA", "DruzynaB"

def _extract_sets_from_scoreboard_box(soup: BeautifulSoup, game_id):
    root = _find_main_game_node(soup, game_id)
    scores = root.select(".game-score")
    if len(scores) < 2:
        return None, None

    a_txt = scores[0].get_text(" ", strip=True)
    b_txt = scores[1].get_text(" ", strip=True)
    if not a_txt.isdigit() or not b_txt.isdigit():
        return None, None

    a = int(a_txt)
    b = int(b_txt)
    if max(a, b) == 3 and (a + b) in (3, 4, 5):
        return a, b

    return None, None

def _extract_sets_from_progress_table(soup: BeautifulSoup):
    """Zwraca (setyA, setyB) z tabeli przebiegu (kolumna 'Wynik', np. 3 : 0)."""
    for tbl in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        if not headers:
            continue
        if "set" not in headers or not any("wynik" in h for h in headers):
            continue

        wynik_idx = None
        for i, h in enumerate(headers):
            if "wynik" in h:
                wynik_idx = i
                break
        if wynik_idx is None:
            continue

        tfoot = tbl.find("tfoot")
        if tfoot:
            tds = tfoot.find_all("td")
            if len(tds) > wynik_idx:
                txt = tds[wynik_idx].get_text(" ", strip=True)
                m = re.search(r"([0-3])\s*:\s*([0-3])", txt)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    if max(a, b) == 3 and (a + b) in (3, 4, 5):
                        return a, b

        tbody = tbl.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
            if rows:
                last_tds = rows[-1].find_all("td")
                if len(last_tds) > wynik_idx:
                    txt = last_tds[wynik_idx].get_text(" ", strip=True)
                    m = re.search(r"([0-3])\s*:\s*([0-3])", txt)
                    if m:
                        a, b = int(m.group(1)), int(m.group(2))
                        if max(a, b) == 3 and (a + b) in (3, 4, 5):
                            return a, b

    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# KROK 1 — lista ID meczów
# ─────────────────────────────────────────────────────────────────────────────
def pobierz_id_meczow(tour_id, sezon):
    url = f"{BASE_URL}/games/tour/{tour_id}.html"
    print(f"\n{'='*60}\nSezon {sezon}  (tour {tour_id})\n{'='*60}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # Szukamy ID meczów — najpierw z /tour/, potem bez (nowsze strony)
    ids = sorted(set(re.findall(rf'/games/action/show/id/(\d+)/tour/{tour_id}', resp.text)))
    if not ids:
        # Fallback: nowsze strony nie mają /tour/ w linkach
        ids = sorted(set(re.findall(r'/games/action/show/id/(\d+)\.html', resp.text)))
        # Filtrujemy tylko ID z tego sezonu (opcjonalne, ale bezpieczne)
    print(f"Znaleziono {len(ids)} ID meczów.")
    return ids

# ─────────────────────────────────────────────────────────────────────────────
# KROK 2a — parsowanie PDF (pełne staty + scoreboard)
# ─────────────────────────────────────────────────────────────────────────────
def parsuj_pdf(pdf_bytes, game_id, sezon):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return None, f"PDF read error: {e}"

    # Jeśli PDF jest obrazem (brak warstwy tekstowej) -> nie parsujemy tutaj
    if not text.strip():
        return None, "PDF bez warstwy tekstowej (extract_text pusty)"

    # Normalizacje:
    text = text.replace("–", "-").replace("−", "-")
    text = re.sub(r"(?<=\s)\.(?=\s)", "0", text)

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── Drużyny + sety ───────────────────────────────────────────────────────
    team1 = team2 = sets1 = sets2 = None
    for line in lines[:50]:
        m = re.match(r"^(.+?)\s+(\d)\s+(\d)\s+(.+)$", line)
        if m:
            team1 = m.group(1).strip()
            sets1 = int(m.group(2))
            sets2 = int(m.group(3))
            team2 = m.group(4).strip()
            break
    if team1 is None:
        return None, "brak wyniku (setów) w PDF"

    n_sets = sets1 + sets2

    # ── Scoreboard per set (odporny na sklejone linie) ───────────────────────
    set_scores_map = {}  # {set_no: (a,b)}
    for l in lines:
        m = re.search(r"\b([1-5])\s+(\d{1,2}:\d{2})\b(.*)$", l)
        if not m:
            continue
        set_no = int(m.group(1))
        tail = m.group(3)

        pairs = re.findall(r"(\d{1,2})\s*-\s*(\d{1,2})", tail)
        if not pairs:
            continue
        a, b = map(int, pairs[-1])
        if a < 10 and b < 10:
            continue
        set_scores_map[set_no] = (a, b)

    set_scores = []
    for sn in range(1, n_sets + 1):
        if sn not in set_scores_map:
            set_scores = None
            break
        set_scores.append(set_scores_map[sn])

    # ── Totalsy drużyn ───────────────────────────────────────────────────────
    pattern = (
        r"(?:Players\s+total|Suma\s+zawodnika|Team\s+total|Suma\s+drużyny|Suma\s+druzyny|Totals?)\s+"
        r"(\d+)\s+(\d+)\s+([+-]?\d+)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)%\s*\((\d+)%\)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+"
        r"(\d+)"
    )
    totals = re.findall(pattern, text, flags=re.IGNORECASE)
    if len(totals) < 2:
        return None, "brak 2x Players total / Suma zawodnika"

    def tuple_to_stats(t):
        return {
            "pts_suma": int(t[0]),
            "pts_bp": int(t[1]),
            "pts_bilans": int(t[2]),
            "srv_suma": int(t[3]),
            "srv_bledy": int(t[4]),
            "srv_asy": int(t[5]),
            "rec_suma": int(t[6]),
            "rec_bledy": int(t[7]),
            "rec_poz_pct": int(t[8]),
            "rec_perf_pct": int(t[9]),
            "atk_suma": int(t[10]),
            "atk_bledy": int(t[11]),
            "atk_blok": int(t[12]),
            "atk_pkt": int(t[13]),
            "atk_skut_pct": int(t[14]),
            "blk_pkt": int(t[15]),
        }

    t1 = tuple_to_stats(totals[0])
    t2 = tuple_to_stats(totals[1])

    return _build_rows(
        game_id=game_id,
        sezon=sezon,
        team1=team1,
        team2=team2,
        sets1=sets1,
        sets2=sets2,
        s1=t1,
        s2=t2,
        source="PDF",
        set_scores=set_scores,
    ), None

# ─────────────────────────────────────────────────────────────────────────────
# KROK 2b — parsowanie HTML pełne (jeśli są statyczne tabele)
# ─────────────────────────────────────────────────────────────────────────────
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

    walkover_markers = ["walkower", "walk over", "w.o.", "walkover",
                        "mecz nie odbył", "brak statystyk", "no stats"]
    if any(m in page_text.lower() for m in walkover_markers):
        return None, "walkower/brak statystyk"

    has_stat_tables = any(
        "Suma" in [th.get_text(strip=True) for th in t.find_all("th")]
        and "Błąd" in [th.get_text(strip=True) for th in t.find_all("th")]
        for t in soup.find_all("table")
    )
    if not has_stat_tables:
        snippet = " ".join(page_text.split())[:200]
        return None, f"brak tabel statystyk — strona: {snippet!r}"

    # wynik setów
    sets1 = sets2 = None
    score_block = soup.find("div", class_=re.compile(r"result|score|wynik", re.I))
    score_text = score_block.get_text() if score_block else page_text
    sm = re.search(r"\b([0-3])\s*:\s*([0-3])\b", score_text)
    if sm:
        a, b = int(sm.group(1)), int(sm.group(2))
        if max(a, b) == 3 and (a + b) in (3, 4, 5):
            sets1, sets2 = a, b

    if sets1 is None:
        title = soup.find("title")
        if title:
            tm = re.search(r"([0-3])\s*:\s*([0-3])", title.get_text())
            if tm:
                a, b = int(tm.group(1)), int(tm.group(2))
                if max(a, b) == 3 and (a + b) in (3, 4, 5):
                    sets1, sets2 = a, b

    if sets1 is None:
        sets1, sets2 = _extract_sets_from_progress_table(soup)

    if sets1 is None:
        return None, "brak wyniku w HTML (tabele są, ale nie znaleziono wyniku)"

    n_sets = sets1 + sets2
    set_scores_all = _extract_set_scores_from_html(soup)
    if not set_scores_all:
        set_scores_all = _extract_set_scores_from_scoreboard_box(soup, game_id)
    set_scores = set_scores_all[:n_sets] if len(set_scores_all) >= n_sets else None

    # nazwy drużyn
    h3s = [h.get_text(strip=True) for h in soup.find_all("h3") if h.get_text(strip=True)]
    team_names = [n for n in h3s if len(n) > 3
                  and "Statystyki" not in n and "Legenda" not in n
                  and "Przebieg" not in n and "Szczeg" not in n]

    STAT_RIDX = {
        "pts_suma":  -21,
        "pts_bp":    -20,
        "pts_bilans":-19,
        "srv_suma":  -18,
        "srv_bledy": -17,
        "srv_asy":   -16,
        "rec_suma":  -14,
        "rec_bledy": -13,
        "rec_poz_pct":  -12,
        "rec_perf_pct": -11,
        "atk_suma":  -10,
        "atk_bledy":  -9,
        "atk_blok":   -8,
        "atk_pkt":    -7,
        "blk_pkt":    -4,
    }
    N_STAT_COLS = 21
    PCT_FROM_TOTAL = {"rec_poz_pct", "rec_perf_pct"}

    def _has_stat_headers(tbl):
        th_texts = {th.get_text(strip=True) for th in tbl.find_all("th")}
        return "Suma" in th_texts and "Błąd" in th_texts

    def _sum_table(tbl):
        if not _has_stat_headers(tbl):
            return None

        totals = {c: 0 for c in STAT_COLS}
        player_count = 0
        SKIP_KW = ["Suma z", "Punkty", "Zagrywka", "total", "zawodnika",
                   "Players total", "Suma zawodnika"]

        rec_pos_sum  = 0.0
        rec_perf_sum = 0.0

        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            vals = [td.get_text(strip=True) for td in tds]
            first = vals[0] if vals else ""

            if any(kw.lower() in first.lower() for kw in SKIP_KW):
                continue
            if len(vals) < N_STAT_COLS + 1:
                continue

            def rv(ridx):
                return _safe_int(vals[ridx]) if len(vals) >= abs(ridx) else 0

            played = (rv(-21) != 0 or rv(-18) > 0 or rv(-10) > 0
                      or rv(-19) != 0 or rv(-14) > 0)
            if not played:
                continue

            for stat, ridx in STAT_RIDX.items():
                if stat in PCT_FROM_TOTAL:
                    continue
                if len(vals) >= abs(ridx):
                    totals[stat] += _safe_int(vals[ridx])

            rec = rv(-14)
            if rec > 0:
                poz_str  = vals[-12] if len(vals) >= 12 else "0"
                perf_str = vals[-11] if len(vals) >= 11 else "0"
                rec_pos_sum  += _safe_int(poz_str)  * rec
                rec_perf_sum += _safe_int(perf_str) * rec

            player_count += 1

        if totals["rec_suma"] > 0:
            totals["rec_poz_pct"]  = round(rec_pos_sum  / totals["rec_suma"])
            totals["rec_perf_pct"] = round(rec_perf_sum / totals["rec_suma"])

        return totals if player_count > 0 else None

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
        ts["atk_skut_pct"] = (round(ts["atk_pkt"] / ts["atk_suma"] * 100)
                              if ts["atk_suma"] > 0 else 0)
        team_stats.append(ts)

    t1 = team_names[0] if len(team_names) > 0 else "Druzyna1"
    t2 = team_names[1] if len(team_names) > 1 else "Druzyna2"

    return _build_rows(
        game_id, sezon, t1, t2, sets1, sets2,
        team_stats[0], team_stats[1], "HTML",
        set_scores=set_scores,
    ), None

# ─────────────────────────────────────────────────────────────────────────────
# HTML fallback: tylko scoreboard (gdy brak tabel statystyk w HTML)
# ─────────────────────────────────────────────────────────────────────────────
def parsuj_html_scoreboard_only(html, game_id, sezon):
    soup = BeautifulSoup(html, "html.parser")

    team1, team2 = _extract_teams_from_match_box(soup, game_id)

    sets1, sets2 = _extract_sets_from_progress_table(soup)
    if sets1 is None:
        sets1, sets2 = _extract_sets_from_scoreboard_box(soup, game_id)
    if sets1 is None:
        return None, "brak przebiegu meczu i brak scoreboardu setów w HTML"

    n_sets = sets1 + sets2
    set_scores_all = _extract_set_scores_from_html(soup)
    if not set_scores_all:
        set_scores_all = _extract_set_scores_from_scoreboard_box(soup, game_id)
    set_scores = set_scores_all[:n_sets] if len(set_scores_all) >= n_sets else None

    empty_stats = {k: "" for k in STAT_COLS}

    return _build_rows(
        game_id=game_id,
        sezon=sezon,
        team1=team1,
        team2=team2,
        sets1=sets1,
        sets2=sets2,
        s1=empty_stats,
        s2=empty_stats,
        source="HTML_SCOREBOARD_ONLY",
        set_scores=set_scores,
    ), None

# ─────────────────────────────────────────────────────────────────────────────
# Budowanie wiersza CSV
# ─────────────────────────────────────────────────────────────────────────────
def _build_rows(game_id, sezon, team1, team2, sets1, sets2, s1, s2, source, set_scores=None):
    row = {
        "game_id": game_id,
        "sezon": sezon,
        "source": source,
        "druzyna_A": team1,
        "druzyna_B": team2,
        "sety_A": sets1,
        "sety_B": sets2,
        "sets_played": sets1 + sets2,
        "wygrana_A": 1 if sets1 > sets2 else 0,
    }

    row.update(_set_scores_to_row_fields(set_scores))

    for k, v in s1.items():
        row[f"A_{k}"] = v
    for k, v in s2.items():
        row[f"B_{k}"] = v

    # diff tylko gdy to liczby (dla HTML_SCOREBOARD_ONLY będą puste)
    for k in s1:
        row[f"diff_{k}"] = (s1[k] - s2[k]) if isinstance(s1[k], int) and isinstance(s2[k], int) else ""

    for fn in FIELDNAMES:
        row.setdefault(fn, "")

    return [row]

# ─────────────────────────────────────────────────────────────────────────────
# KROK 3 — PDF → HTML fallback
# ─────────────────────────────────────────────────────────────────────────────
def pobierz_mecz(game_id, sezon, tour_id):
    pdf_err = "nie próbowano"

    # 1) próba PDF
    try:
        r = requests.get(
            f"{BASE_URL}/games/action/stats/id/{game_id}/tour/{tour_id}.html",
            headers=HEADERS,
            timeout=30,
        )
        ct = r.headers.get("Content-Type", "")
        if r.status_code == 200 and ("pdf" in ct.lower() or r.content[:4] == b"%PDF"):
            rows, err = parsuj_pdf(r.content, game_id, sezon)
            if rows:
                return rows, "PDF", None
            pdf_err = err or "PDF parse failed"
        else:
            pdf_err = f"brak PDF (status={r.status_code}, ct={ct[:40]})"
    except Exception as e:
        pdf_err = str(e)

    # 2) fallback HTML
    try:
        r = requests.get(
            f"{BASE_URL}/games/action/show/id/{game_id}/tour/{tour_id}.html",
            headers=HEADERS,
            timeout=30,
        )
        if r.status_code != 200:
            return None, None, f"HTML HTTP {r.status_code}"

        # 2a) pełne statystyki (jeśli statyczne tabele są w HTML)
        rows, err = parsuj_html(r.text, game_id, sezon)
        if rows:
            return rows, "HTML", None

        # 2b) ostatecznie: tylko scoreboard
        rows2, err2 = parsuj_html_scoreboard_only(r.text, game_id, sezon)
        if rows2:
            return rows2, "HTML_SCOREBOARD_ONLY", None

        return None, None, f"PDF({pdf_err}) | HTML({err}) | HTML_SCOREBOARD_ONLY({err2})"
    except Exception as e:
        return None, None, f"PDF({pdf_err}) | HTML exception: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA PĘTLA (zapis przyrostowy + resume)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Wczytaj już pobrane ID (resume)
    istniejace_ids = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gid = (row.get("game_id") or "").strip()
                if gid:
                    istniejace_ids.add(gid)
        if istniejace_ids:
            print(f"Plik istnieje — {len(istniejace_ids)} meczów już w bazie, wznawiam.")

    bledy = []

    # Otwórz plik CSV w trybie append
    plik_istnieje = os.path.exists(OUTPUT_CSV)
    f_out = open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if not plik_istnieje:
        w.writeheader()
        f_out.flush()

    for sezon, tour_id in SEZONY.items():
        ids = pobierz_id_meczow(tour_id, sezon)
        time.sleep(DELAY)

        for i, gid in enumerate(ids, 1):
            gid_str = str(gid)

            # Resume: pomiń już pobrane
            if gid_str in istniejace_ids:
                print(f"[{sezon}] {i:3d}/{len(ids)} ID={gid} ... [JUŻ JEST — pomijam]")
                continue

            print(f"[{sezon}] {i:3d}/{len(ids)} ID={gid} ...", end=" ", flush=True)
            rows, source, err = pobierz_mecz(gid, sezon, tour_id)

            if rows:
                row = rows[0]
                sb = ""
                if row.get("A_scoreboard_points") != "" and row.get("B_scoreboard_points") != "":
                    sb = f" | SB {row['A_scoreboard_points']}:{row['B_scoreboard_points']}"
                print(f"[{source}] {row['druzyna_A'][:18]} {row['sety_A']}:{row['sety_B']} {row['druzyna_B'][:18]}{sb}")
                # Zapisz natychmiast
                w.writerow(row)
                f_out.flush()
                istniejace_ids.add(gid_str)
            else:
                print(f"SKIP — {err}")
                bledy.append({"sezon": sezon, "game_id": gid, "blad": err})

            time.sleep(DELAY)

    f_out.close()

    # Błędy
    if bledy:
        bledy_istnieja = os.path.exists("tauron_bledy.csv")
        with open("tauron_bledy.csv", "a", newline="", encoding="utf-8-sig") as f:
            wb = csv.DictWriter(f, fieldnames=["sezon","game_id","blad"])
            if not bledy_istnieja:
                wb.writeheader()
            wb.writerows(bledy)

    # Podsumowanie
    print()
    with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
        total = sum(1 for _ in csv.DictReader(f))
    print("="*60)
    print(f"GOTOWE: {total} meczów → {OUTPUT_CSV}")
    if bledy:
        print(f"Pominięte: {len(bledy)} → tauron_bledy.csv")
    print("="*60)

if __name__ == "__main__":
    main()
