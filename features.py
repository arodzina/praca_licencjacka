# build_master_and_splits.py
# Czyta: data/tauron_liga_statystyki_final.csv
# Tworzy:
#   - df_master   (pełny, do EDA/debug/aneks)
#   - df_postmatch (CLEAN: ID + (czy_playoff) + target + diff_*)
#   - df_prematch (tylko cechy historyczne + h2h)

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
    "EcoHarpoon NOWEL LOS No…":"EcoHarpoon NOWEL LOS Nowy Dwór Mazowiecki",
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
# 3) TEAM HISTORY (PRE-MATCH) - NO LEAKAGE
# =========================
def add_season_progress_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "game_id"]).reset_index(drop=True).copy()

    df["match_order"] = df.groupby("season").cumcount()
    season_n = df.groupby("season")["game_id"].transform("count")
    df["season_progress"] = df["match_order"] / season_n  # 0..(1 - 1/n)

    return df
def add_team_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "game_id"]).reset_index(drop=True).copy()

    team_a = pd.DataFrame({
        "season": df["season"],
        "game_id": df["game_id"],
        "team": df["team_A"],
        "win": df[TARGET],
        "pts": df["A_pts_total"],
        "errors": df["A_errors_total"],
        "atk_eff": df["A_atk_eff"],
    })
    team_b = pd.DataFrame({
        "season": df["season"],
        "game_id": df["game_id"],
        "team": df["team_B"],
        "win": 1 - df[TARGET],
        "pts": df["B_pts_total"],
        "errors": df["B_errors_total"],
        "atk_eff": df["B_atk_eff"],
    })

    teams = pd.concat([team_a, team_b], ignore_index=True)
    teams = teams.sort_values(["season", "team", "game_id"]).reset_index(drop=True)

    g = teams.groupby(["season", "team"])

    teams["matches_before"] = g.cumcount()
    teams["win_rate_prev"] = g["win"].transform(lambda s: s.shift(1).expanding().mean())
    teams["win_rate_last5"] = g["win"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    teams["pts_prev_all"] = g["pts"].transform(lambda s: s.shift(1).expanding().mean())
    teams["errors_prev_all"] = g["errors"].transform(lambda s: s.shift(1).expanding().mean())
    teams["atk_eff_prev_all"] = g["atk_eff"].transform(lambda s: s.shift(1).expanding().mean())

    # liczba wygranych w ostatnich 3 (nie klasyczny streak)
    teams["wins_last3"] = g["win"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())

    teams["pts_rolling3"] = g["pts"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    teams["errors_rolling3"] = g["errors"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    feat = teams[[
        "season", "game_id", "team",
        "matches_before", "win_rate_prev", "win_rate_last5",
        "pts_prev_all", "errors_prev_all", "atk_eff_prev_all",
        "wins_last3", "pts_rolling3", "errors_rolling3",
    ]].copy()

    a = feat.rename(columns={"team": "team_A"})
    b = feat.rename(columns={"team": "team_B"})

    df2 = df.merge(a, on=["season", "game_id", "team_A"], how="left", suffixes=("", "_A"))
    df2 = df2.merge(b, on=["season", "game_id", "team_B"], how="left", suffixes=("_A", "_B"))

    hist_cols = [
        "matches_before", "win_rate_prev", "win_rate_last5",
        "pts_prev_all", "errors_prev_all", "atk_eff_prev_all",
        "wins_last3", "pts_rolling3", "errors_rolling3",
    ]
    for col in hist_cols:
        df2[f"diff_{col}"] = df2[f"{col}_A"] - df2[f"{col}_B"]

    return df2

# =========================
# 4) H2H (PRE-MATCH SAFE)
# =========================
def add_czy_playoff_from_h2h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "game_id"]).reset_index(drop=True).copy()

    # jeśli nie ma h2h_matches_before, policz ją
    if "h2h_matches_before" not in df.columns:
        df["h2h_pair"] = df.apply(lambda r: tuple(sorted((r["team_A"], r["team_B"]))), axis=1)
        df["h2h_matches_before"] = df.groupby(["season", "h2h_pair"]).cumcount()
        df = df.drop(columns=["h2h_pair"])

    # playoff = 1, gdy to co najmniej 3. mecz tych drużyn w sezonie (po 2 meczach zasadniczych)
    df["czy_playoff"] = (df["h2h_matches_before"] >= 2).astype(int)
    return df
def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "game_id"]).reset_index(drop=True).copy()

    df["h2h_pair"] = df.apply(lambda r: tuple(sorted((r["team_A"], r["team_B"]))), axis=1)
    g = df.groupby(["season", "h2h_pair"])

    df["h2h_matches_before"] = g.cumcount()
    df["h2h_win_rate_A"] = g[TARGET].transform(lambda s: s.shift(1).expanding().mean())
    df["h2h_last_result_A"] = g[TARGET].transform(lambda s: s.shift(1)).fillna(0.5)

    return df.drop(columns=["h2h_pair"])

# =========================
# 5) SPLIT
# =========================
def split_postmatch_prematch(df_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # PREMATCH: historia + h2h + target + identyfikatory
    prematch_cols = [
        "game_id", "season", "team_A", "team_B", TARGET,
        "czy_playoff",
        "h2h_matches_before", "h2h_win_rate_A", "h2h_last_result_A",
        "matches_before_A", "win_rate_prev_A", "win_rate_last5_A",
        "pts_prev_all_A", "errors_prev_all_A", "atk_eff_prev_all_A",
        "wins_last3_A", "pts_rolling3_A", "errors_rolling3_A",
        "matches_before_B", "win_rate_prev_B", "win_rate_last5_B",
        "pts_prev_all_B", "errors_prev_all_B", "atk_eff_prev_all_B",
        "wins_last3_B", "pts_rolling3_B", "errors_rolling3_B",
        "diff_matches_before", "diff_win_rate_prev", "diff_win_rate_last5",
        "diff_pts_prev_all", "diff_errors_prev_all", "diff_atk_eff_prev_all",
        "diff_wins_last3", "diff_pts_rolling3", "diff_errors_rolling3", "match_order", "season_progress",
    ]
    prematch_cols = [c for c in prematch_cols if c in df_master.columns]
    df_prematch = df_master[prematch_cols].copy()
    # POSTMATCH: ID + kontekst + target + diff-y z meczu
    postmatch_cols = [
        # ID / kontekst
        "game_id",
        "season",
        "team_A",
        "team_B",
        "number_of_sets",  # potrzebne do tie-breaków
        "czy_playoff",  # dodasz wg reguły H2H > 2 w sezonie

        # target
        "win_A",

        # --- różnice z meczu (część "surowa")
        "diff_atk_errors",
        "diff_atk_blocked",
        "diff_atk_points",
        "diff_atk_success_pct",

        "diff_srv_aces",
        "diff_srv_errors",

        "diff_rec_perf_pct",
        "diff_rec_poz_pct",
        "diff_rec_errors",

        "diff_blk_points",

        # --- cechy utworzone przez Ciebie (feature engineering)
        "diff_atk_eff",
        "diff_srv_eff",
        "diff_errors_total",
        "diff_opp_errors_share",
        "diff_opp_errors_pts",

        # --- cechy relatywne (procentowe) ---
        "diff_rec_error_rate",
        "diff_srv_ace_rate",
        "diff_srv_error_rate",
        "diff_atk_error_rate",
        "diff_atk_blocked_rate",
        "diff_blk_per_set",
        "diff_error_total_rate",
    ]
    postmatch_cols = [c for c in postmatch_cols if c in df_master.columns]
    df_postmatch = df_master[postmatch_cols].copy()




    return df_postmatch, df_prematch

# =========================
# MAIN PIPELINE
# =========================
def build_all(input_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_raw = load_final(input_file)
    df0 = rename_and_types(df_raw)
    df0 = df0.sort_values(["season", "game_id"]).reset_index(drop=True)
    df0 = add_season_progress_features(df0)

    df1 = add_match_features(df0)
    df2 = add_team_history_features(df1)

    df3 = add_h2h_features(df2)
    df3 = add_czy_playoff_from_h2h(df3)

    df_master = df3.drop(columns=DROP_ALWAYS, errors="ignore")
    df_postmatch, df_prematch = split_postmatch_prematch(df_master)

    return df_master, df_postmatch, df_prematch

if __name__ == "__main__":
    df_master, df_postmatch, df_prematch = build_all(INPUT_FILE)
    print("MASTER:", df_master.shape)
    print("POSTMATCH:", df_postmatch.shape)
    print("PREMATCH:", df_prematch.shape)
    df_master.to_csv("data/processed_master.csv", index=False, sep=",", encoding="utf-8-sig")
    df_postmatch.to_csv("data/processed_postmatch.csv", index=False, sep=",", encoding="utf-8-sig")
    df_prematch.to_csv("data/processed_prematch.csv", index=False, sep=",", encoding="utf-8-sig")
    print(df_postmatch['season'])
    print(df_prematch.head())