# cechy.py
# Czyta: data/tauron_liga_statystyki_final.csv
# Tworzy:
#   - df_master   (pełny, do EDA/debug/aneks)
#   - df_postmatch (CLEAN: ID + (is_playoff) + target + diff_*)

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_FILE = Path("data/tauron_liga_statystyki_final.csv")
SEP = ";"

TARGET_RAW = "wygrana_A"
TARGET = "win_A"

DROP_ALWAYS = ["druzyna_A", "druzyna_B", "source"]  # do wyrzucenia po zbudowaniu team_A/team_B

TEAM_MAP = {
    "KS PAŁAC Bydgoszcz": "Pałac Bydgoszcz",
    "Bank Pocztowy Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "Polskie Przetwory Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "OnlyBio Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "Metalkas Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "Metalkas PAŁAC Bydgoszcz": "Pałac Bydgoszcz",
    "Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "KS Pałac Bydgoszcz": "Pałac Bydgoszcz",
    "Polskie Przetwory Palac Byd…": "Pałac Bydgoszcz",
    "Polskie Przetwory Pałac Byd…": "Pałac Bydgoszcz",
    "Chemik Police": "Chemik Police",
    "Grupa Azoty Chemik Police": "Chemik Police",
    "LOTTO Chemik Police": "Chemik Police",
    "Developres SkyRes Rzeszów": "Developres Rzeszów",
    "KS Developres Rzeszów": "Developres Rzeszów",
    "KS DevelopRes Rzeszów": "Developres Rzeszów",
    "Developres BELLA DOLINA Rzeszów": "Developres Rzeszów",
    "Developres BELLA DOLINA R…": "Developres Rzeszów",
    "PGE RYSICE Rzeszów": "Developres Rzeszów",
    "PGE Rysice Rzeszów": "Developres Rzeszów",
    "Grot Budowlani Łódź": "Budowlani Łódź",
    "PGE Grot Budowlani Łódź": "Budowlani Łódź",
    "Grot Budowlani Lódz": "Budowlani Łódź",
    "Budowlani Łódź": "Budowlani Łódź",
    "PGE Budowlani Łódź": "Budowlani Łódź",
    "ŁKS Commercecon Łódź": "ŁKS Commercecon Łódź",
    "LKS Commercecon Lódz": "ŁKS Commercecon Łódź",
    "ŁKS Commerceon Łódź": "ŁKS Commercecon Łódź",
    "ŁKS COMMERCECON ŁÓDŹ":"ŁKS Commercecon Łódź",
    "#VolleyWrocław": "Volley Wrocław",
    "#VolleyWroclaw": "Volley Wrocław",
    "KGHM #VolleyWrocław": "Volley Wrocław",
    "#Volley Wrocław": "Volley Wrocław",
    "Impel Wrocław": "Volley Wrocław",
    "IŁ Capital Legionovia Legion…": "Legionovia",
    "DPD IŁCapital Legionovia Le…": "Legionovia",
    "Legionovia Legionowo": "Legionovia",
    "DPD Legionovia Legionowo": "Legionovia",
    "IŁ Capital Legionovia Legionowo": "Legionovia",
    "IL Capital Legionovia Legion…": "Legionovia",
    "DPD IŁCapital Legionovia Legionowo": "Legionovia",
    "DPD ILCapital Legionovia Le…": "Legionovia",
    "E.Leclerc Radomka Radom": "Radomka Radom",
    "E.LECLERC Radomka Radom": "Radomka Radom",
    "E.LECLERC MOYA Radomka …": "Radomka Radom",
    "E.LECLERC MOYA Radomka Radom": "Radomka Radom",
    "MOYA Radomka Radom": "Radomka Radom",
    "MOYA Radomka Lotnisko Radom": "Radomka Radom",
    "MOYA Radomka Lotnisko Ra…": "Radomka Radom",
    "ITA TOOLS STAL Mielec": "Stal Mielec",
    "ITA TOOLS Stal Mielec": "Stal Mielec",
    "ITA TOOLS  STAL Mielec": "Stal Mielec",
    "ITA TOOLS  STAL Mielec": "Stal Mielec",
    "Grupa Azoty Akademia Tarnów": "Akademia Tarnów",
    "Grupa Azoty Akademia Tarnó…": "Akademia Tarnów",
    "ROLESKI GRUPA AZOTY Tarnów": "Akademia Tarnów",
    "ROLESKI GRUPA AZOTY Tarn…": "Akademia Tarnów",
    "BKS PROFI CREDIT Bielsko-Biała": "BKS Bielsko-Biała",
    "BKS STAL Bielsko-Biała": "BKS Bielsko-Biała",
    "BKS STAL Bielsko-Biala": "BKS Bielsko-Biała",
    "BKS BOSTIK Bielsko-Biała": "BKS Bielsko-Biała",
    "BKS BOSTIK ZGO Bielsko-Biała": "BKS Bielsko-Biała",
    "BKS BOSTIK ZGO Bielsko-Bia…": "BKS Bielsko-Biała",
    "BKS ALUPROF PROFI CREDIT Bielsko-Biała": "BKS Bielsko-Biała",
    "KSZO OSTROWIEC": "KSZO Ostrowiec",
    "KSZO Ostrowiec": "KSZO Ostrowiec",
    "KSZO Ostrowiec Świętokrzyski": "KSZO Ostrowiec",
    "Enea PTPS Piła": "PTPS Piła",
    "PTPS Piła": "PTPS Piła",
    "Energa MKS Kalisz": "MKS Kalisz",
    "MKS Kalisz": "MKS Kalisz",
    "UNI Opole": "UNI Opole",
    "Joker Świecie": "Joker Świecie",
    "Joker Swiecie": "Joker Świecie",
    "Sokół & Hagric Mogilno": "Sokół Mogilno",
    "Wisła Warszawa": "Wisła Warszawa",
    "Enea PTPS Pila": "PTPS Piła",
    "Atom Trefl Sopot": "Atom Trefl Sopot",
    "PGE Atom Trefl Sopot": "Atom Trefl Sopot",
    "Polski Cukier Muszynianka": "Muszynianka Muszyna",
    "Polski Cukier Muszynianka Enea": "Muszynianka Muszyna",
    "Polski Cukier Muszynianka Muszyna": "Muszynianka Muszyna",
    "Giacomini Budowlani Toruń": "Budowlani Toruń",
    "POLI Budowlani Toruń": "Budowlani Toruń",
    "Trefl Proxima Kraków": "Proxima Kraków",
    "Tauron MKS Dąbrowa Górnicza": "MKS Dąbrowa Górnicza",
    "MKS Dąbrowa Górnicza": "MKS Dąbrowa Górnicza",
    "EcoHarpoon NOWEL LOS No…":"LOS Nowy Dwór Mazowiecki",
    "EcoHarpoon NOWEL LOS Nowy Dwór Mazowiecki":"LOS Nowy Dwór Mazowiecki",
    "Sokol & Hagric Mogilno":"Sokół Mogilno"
}

RENAME_MAP = {
    # meta
    "sezon": "season",
    "sety_A": "sets_A",
    "sety_B": "sets_B",
    "wygrana_A": "win_A",

    # A
    "A_pts_suma": "A_pts_total",
    "A_pts_bp": "A_pts_bp",
    "A_pts_bilans": "A_pts_balance",
    "A_srv_suma": "A_srv_total",
    "A_srv_bledy": "A_srv_errors",
    "A_srv_asy": "A_srv_aces",
    "A_rec_suma": "A_rec_total",
    "A_rec_bledy": "A_rec_errors",
    "A_rec_poz_pct": "A_rec_pos_pct",
    "A_rec_perf_pct": "A_rec_perf_pct",
    "A_atk_suma": "A_atk_total",
    "A_atk_bledy": "A_atk_errors",
    "A_atk_blok": "A_atk_blocked",
    "A_atk_pkt": "A_atk_points",
    "A_atk_skut_pct": "A_atk_success_pct",
    "A_blk_pkt": "A_blk_points",

    # B
    "B_pts_suma": "B_pts_total",
    "B_pts_bp": "B_pts_bp",
    "B_pts_bilans": "B_pts_balance",
    "B_srv_suma": "B_srv_total",
    "B_srv_bledy": "B_srv_errors",
    "B_srv_asy": "B_srv_aces",
    "B_rec_suma": "B_rec_total",
    "B_rec_bledy": "B_rec_errors",
    "B_rec_poz_pct": "B_rec_pos_pct",
    "B_rec_perf_pct": "B_rec_perf_pct",
    "B_atk_suma": "B_atk_total",
    "B_atk_bledy": "B_atk_errors",
    "B_atk_blok": "B_atk_blocked",
    "B_atk_pkt": "B_atk_points",
    "B_atk_skut_pct": "B_atk_success_pct",
    "B_blk_pkt": "B_blk_points",

    # diff (PL -> EN)
    "diff_pts_suma": "diff_pts_total",
    "diff_pts_bilans": "diff_pts_balance",
    "diff_srv_suma": "diff_srv_total",
    "diff_srv_bledy": "diff_srv_errors",
    "diff_srv_asy": "diff_srv_aces",
    "diff_rec_suma": "diff_rec_total",
    "diff_rec_bledy": "diff_rec_errors",
    "diff_atk_suma": "diff_atk_total",
    "diff_atk_bledy": "diff_atk_errors",
    "diff_atk_blok": "diff_atk_blocked",
    "diff_atk_pkt": "diff_atk_points",
    "diff_atk_skut_pct": "diff_atk_success_pct",
    "diff_blk_pkt": "diff_blk_points",
}

# =========================
# 0) LOAD
# =========================
def load_final(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=SEP, encoding="utf-8-sig")

    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df = df.dropna(subset=["game_id"]).copy()
    df["game_id"] = df["game_id"].astype(int)

    # ujednolicone nazwy drużyn (A = gospodarz)
    df["team_A"] = df["druzyna_A"].map(lambda v: TEAM_MAP.get(v, v))
    df["team_B"] = df["druzyna_B"].map(lambda v: TEAM_MAP.get(v, v))

    return df

# =========================
# 1) RENAME + TYPES
# =========================
def rename_and_types(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.rename(columns=RENAME_MAP)

    # target zabezpieczenie
    if TARGET_RAW in df.columns and TARGET not in df.columns:
        df[TARGET] = df[TARGET_RAW]

    # typy
    for c in [ "number_of_sets", "sets_A", "sets_B", "A_scoreboard_points", "B_scoreboard_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================
# 2) MATCH FEATURES (POST-MATCH) – opcjonalne, ale przydatne do EDA / master
# =========================
def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # scoreboard context
    if {"A_scoreboard_points", "B_scoreboard_points"}.issubset(df.columns):
        df["match_total_points"] = df["A_scoreboard_points"] + df["B_scoreboard_points"]

    # efficiencies
    df["A_srv_eff"] = (
        (df["A_srv_aces"] - df["A_srv_errors"]) / df["A_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_srv_eff"] = (
        (df["B_srv_aces"] - df["B_srv_errors"]) / df["B_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_srv_eff"] = df["A_srv_eff"] - df["B_srv_eff"]

    df["A_atk_eff"] = (
        (df["A_atk_points"] - df["A_atk_errors"] - df["A_atk_blocked"])
        / df["A_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_atk_eff"] = (
        (df["B_atk_points"] - df["B_atk_errors"] - df["B_atk_blocked"])
        / df["B_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_atk_eff"] = df["A_atk_eff"] - df["B_atk_eff"]

    # total errors
    df["A_errors_total"] = df["A_srv_errors"] + df["A_atk_errors"] + df["A_rec_errors"]
    df["B_errors_total"] = df["B_srv_errors"] + df["B_atk_errors"] + df["B_rec_errors"]
    df["diff_errors_total"] = df["A_errors_total"] - df["B_errors_total"]

    # ===== CEECHY RELATYWNE (PROCENTOWE) =====
    # zamiast surowych różnic – różnice proporcji/współczynnikóœ

    # --- przyjęcie: procent błędów ---
    df["A_rec_error_rate"] = (
        df["A_rec_errors"] / df["A_rec_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_rec_error_rate"] = (
        df["B_rec_errors"] / df["B_rec_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_rec_error_rate"] = df["A_rec_error_rate"] - df["B_rec_error_rate"]

    # --- zagrywka: procent asów i procent błędów ---
    df["A_srv_ace_rate"] = (
        df["A_srv_aces"] / df["A_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_srv_ace_rate"] = (
        df["B_srv_aces"] / df["B_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_srv_ace_rate"] = df["A_srv_ace_rate"] - df["B_srv_ace_rate"]

    df["A_srv_error_rate"] = (
        df["A_srv_errors"] / df["A_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_srv_error_rate"] = (
        df["B_srv_errors"] / df["B_srv_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_srv_error_rate"] = df["A_srv_error_rate"] - df["B_srv_error_rate"]

    # --- atak: procent błędów i procent zablokowanych ---
    df["A_atk_error_rate"] = (
        df["A_atk_errors"] / df["A_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_atk_error_rate"] = (
        df["B_atk_errors"] / df["B_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_atk_error_rate"] = df["A_atk_error_rate"] - df["B_atk_error_rate"]

    df["A_atk_blocked_rate"] = (
        df["A_atk_blocked"] / df["A_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["B_atk_blocked_rate"] = (
        df["B_atk_blocked"] / df["B_atk_total"].replace(0, np.nan)
    ).fillna(0)
    df["diff_atk_blocked_rate"] = df["A_atk_blocked_rate"] - df["B_atk_blocked_rate"]

    # --- blok: punkty bloku na set (bo drużyny grają różną liczbę setów) ---
    df["A_blk_per_set"] = (
        df["A_blk_points"] / df["number_of_sets"].replace(0, np.nan)
    ).fillna(0)
    df["B_blk_per_set"] = (
        df["B_blk_points"] / df["number_of_sets"].replace(0, np.nan)
    ).fillna(0)
    df["diff_blk_per_set"] = df["A_blk_per_set"] - df["B_blk_per_set"]

    # --- błędy ogółem: procent błędów względem wszystkich akcji ---
    df["A_total_actions"] = df["A_srv_total"] + df["A_atk_total"] + df["A_rec_total"]
    df["B_total_actions"] = df["B_srv_total"] + df["B_atk_total"] + df["B_rec_total"]
    df["A_error_total_rate"] = (
        df["A_errors_total"] / df["A_total_actions"].replace(0, np.nan)
    ).fillna(0)
    df["B_error_total_rate"] = (
        df["B_errors_total"] / df["B_total_actions"].replace(0, np.nan)
    ).fillna(0)
    df["diff_error_total_rate"] = df["A_error_total_rate"] - df["B_error_total_rate"]

    # --- punkty z ataku: procent (już jest diff_atk_success_pct, ale dodajmy też surową różnicę proporcji) ---
    # diff_atk_success_pct już istnieje jako różnica procentowa, OK

    # opponent errors (scoreboard - skill points)
    if {"A_scoreboard_points", "B_scoreboard_points"}.issubset(df.columns):
        df["A_opp_errors_pts"] = df["A_scoreboard_points"] - df["A_pts_total"]
        df["B_opp_errors_pts"] = df["B_scoreboard_points"] - df["B_pts_total"]
        df["diff_opp_errors_pts"] = df["A_opp_errors_pts"] - df["B_opp_errors_pts"]

        df["A_opp_errors_share"] = (
            df["A_opp_errors_pts"] / df["A_scoreboard_points"].replace(0, np.nan)
        ).fillna(0)
        df["B_opp_errors_share"] = (
            df["B_opp_errors_pts"] / df["B_scoreboard_points"].replace(0, np.nan)
        ).fillna(0)
        df["diff_opp_errors_share"] = df["A_opp_errors_share"] - df["B_opp_errors_share"]

    return df

# =========================
# 3) SIDE-OUT EFFICIENCY
# =========================
def add_sideout_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Obliczamy punkty Side-out (Punkty zdobyte, gdy rywal zagrywał)
    # W siatkówce: Scoreboard Points = SideOut Points + Break Points
    df["A_pts_so"] = df["A_scoreboard_points"] - df["A_pts_bp"]
    df["B_pts_so"] = df["B_scoreboard_points"] - df["B_pts_bp"]

    # 2. Side-out Efficiency (Ratio punktów SO do liczby przyjęć)
    df["A_so_eff"] = (df["A_pts_so"] / df["A_rec_total"].replace(0, np.nan)).fillna(0)
    df["B_so_eff"] = (df["B_pts_so"] / df["B_rec_total"].replace(0, np.nan)).fillna(0)

    # 3. Różnica (zmienna do modelu)
    df["diff_so_eff"] = df["A_so_eff"] - df["B_so_eff"]

    return df

# =========================
# 4) WIN-LOSS BALANCE
# =========================
def add_win_loss_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bilans dla drużyny A
    df["A_wl_balance"] = (
        (df["A_srv_aces"] + df["A_atk_points"] + df["A_blk_points"]) -
        (df["A_srv_errors"] + df["A_rec_errors"] + df["A_atk_errors"] + df["A_atk_blocked"])
    )

    # Bilans dla drużyny B
    df["B_wl_balance"] = (
        (df["B_srv_aces"] + df["B_atk_points"] + df["B_blk_points"]) -
        (df["B_srv_errors"] + df["B_rec_errors"] + df["B_atk_errors"] + df["B_atk_blocked"])
    )

    # Normalizacja per set (kluczowa dla rzetelności modelu!)
    df["A_wl_per_set"] = df["A_wl_balance"] / df["number_of_sets"].replace(0, np.nan)
    df["B_wl_per_set"] = df["B_wl_balance"] / df["number_of_sets"].replace(0, np.nan)

    # Różnica (cecha do modelu)
    df["diff_wl_per_set"] = df["A_wl_per_set"] - df["B_wl_per_set"]

    return df

# =========================
# 5) PLAYOFF CONTEXT
# =========================
def add_is_playoff(df: pd.DataFrame) -> pd.DataFrame:
    # Sortowanie chronologiczne: najpierw data, potem game_id dla stabilności
    sort_cols = ["season", "date", "game_id"] if "date" in df.columns else ["season", "game_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True).copy()

    # Liczymy, który to mecz tych dwóch drużyn w danym sezonie
    df["h2h_pair"] = df.apply(lambda r: tuple(sorted((r["team_A"], r["team_B"]))), axis=1)
    df["h2h_matches_before"] = df.groupby(["season", "h2h_pair"]).cumcount()
    df = df.drop(columns=["h2h_pair"])

    # Playoff = co najmniej 3. mecz tych drużyn w sezonie
    df["is_playoff"] = (df["h2h_matches_before"] >= 2).astype(int)
    return df.drop(columns=["h2h_matches_before"])

# =========================
# 6) POSTMATCH SELECTION
# =========================
def select_postmatch_cols(df_master: pd.DataFrame) -> pd.DataFrame:
    # POSTMATCH: ID + kontekst + target + diff-y z meczu
    postmatch_cols = [
        # ===== ID / kontekst =====
        "game_id",
        "season",
        "date",
        "team_A",
        "team_B",
        "number_of_sets",      # Grupa 5: długość meczu
        "is_playoff",         # Grupa 5: faza rozgrywek

        # ===== target =====
        "win_A",

        # ===== Grupa 1: Efektywność Ataku (Siła ognia) =====
        "diff_atk_eff",        # Królowa statystyk ataku
        "diff_so_eff",         # Side-Out Efficiency

        # ===== Grupa 2: System Defensywny (Blok i Obrona) =====
        "diff_blk_per_set",    # Bloki punktowe na set
        "diff_atk_blocked_rate", # Jak często dajemy się zablokować

        # ===== Grupa 3: Zagrywka i Przyjęcie (Inicjacja akcji) =====
        "diff_srv_eff",        # Bilans asów i błędów
        "diff_rec_pos_pct",    # Procent przyjęcia pozytywnego
        "diff_srv_ace_rate",   # Presja zagrywką

        # ===== Grupa 4: Dyscyplina i Błędy Własne (Oddane punkty) =====
        "diff_error_total_rate", # Łączny % błędów własnych
        "diff_opp_errors_share", # % punktów z błędów przeciwnika

        # ===== Win-Loss Balance =====
        "diff_wl_per_set",       # Bilans (atk+blk+srv) - (błędy) na set
    ]
    postmatch_cols = [c for c in postmatch_cols if c in df_master.columns]
    df_postmatch = df_master[postmatch_cols].copy()
    return df_postmatch

# =========================
# MAIN PIPELINE
# =========================
def build_all(input_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = load_final(input_file)
    df0 = rename_and_types(df_raw)

    # Dodanie dat meczów (potrzebne do chronologicznego wyznaczania playoff)
    dates_df = pd.read_csv("data/match_dates.csv", encoding="utf-8-sig")
    df0 = df0.merge(dates_df[["game_id", "date"]], on="game_id", how="left")
    df0["date"] = pd.to_datetime(df0["date"], format="%d.%m.%Y", errors="coerce")

    df0 = df0.sort_values(["season", "date", "game_id"]).reset_index(drop=True)

    df1 = add_match_features(df0)
    df2 = add_sideout_features(df1)
    df3 = add_win_loss_features(df2)
    df4 = add_is_playoff(df3)
    df_master = df4.drop(columns=DROP_ALWAYS, errors="ignore")
    df_postmatch = select_postmatch_cols(df_master)

    return df_master, df_postmatch

if __name__ == "__main__":
    df_master, df_postmatch = build_all(INPUT_FILE)
    print("MASTER:", df_master.shape)
    print("POSTMATCH:", df_postmatch.shape)
    df_master.to_csv("data/processed_master.csv", index=False, sep=",", encoding="utf-8-sig")
    df_postmatch.to_csv("data/processed_postmatch.csv", index=False, sep=",", encoding="utf-8-sig")
    print(df_postmatch['season'])