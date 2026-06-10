# Analiza statystyk meczowych TAURON Ligi Kobiet

Kod źródłowy do pracy licencjackiej dotyczącej wpływu statystyk meczowych na wynik spotkań w TAURON Lidze Kobiet w piłce siatkowej.

## Struktura

```
.
├── analiza_i_modelowanie.ipynb    # Główny notebook z analizą i modelowaniem
├── cechy.py                       # Budowa cech (krok 5)
├── scraper/                        # Skrypty do zbierania danych
│   ├── scraper_tauron.py          # (1) Główny scraper (PDF → HTML → scoreboard)
│   ├── 01_pobierz_wyniki_setow.py # (2) Punkty setowe
│   ├── 02_scal_wyniki_setow.py    # (3) Scalanie scoreboard
│   ├── 03_usun_typ_meczu_dodaj_liczbe_setow.py  # (4) Finalne czyszczenie
│   ├── 04_pobierz_daty_meczow.py  # Daty meczów
│   └── scal_wyniki_z_danymi.py
├── data/                            # Dane (zobacz .gitignore)
│   ├── processed_postmatch.csv      # Główny zbiór do modelowania
│   └── match_dates.csv
├── requirements.txt                 # Zależności
└── .gitignore
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie (kolejność)

1. **Pobranie surowych statystyk:** `python scraper/scraper_tauron.py`
2. **Pobranie punktów setowych:** `python scraper/01_pobierz_wyniki_setow.py`
3. **Scalenie scoreboard:** `python scraper/02_scal_wyniki_setow.py`
4. **Finalne czyszczenie:** `python scraper/03_usun_typ_meczu_dodaj_liczbe_setow.py`
5. **Budowa cech:** `python cechy.py`
6. **Analiza i modelowanie:** otwórz `analiza_i_modelowanie.ipynb` w Jupyter

> **Uwaga:** Plik `data/processed_postmatch.csv` (wynik kroku 5) znajduje się już w repozytorium,
> więc do samego odtworzenia analizy wystarczy krok 6.

## Uwagi

- Dane pochodzą ze strony [tauronliga.pl](https://www.tauronliga.pl)
- Analiza ma charakter *ex post* — bada związek statystyk meczowych z wynikiem
- Szczegółowy opis metodologii i wyników znajduje się w pracy licencjackiej
