# Analiza statystyk meczowych TAURON Ligi Kobiet

Kod źródłowy do pracy licencjackiej dotyczącej wpływu statystyk meczowych na wynik spotkań w TAURON Lidze Kobiet w piłce siatkowej.

## Struktura

```
.
├── analiza_i_modelowanie.ipynb       # Główny notebook z analizą i modelowaniem
├── cechy.py                          # Definicje cech
├── scraper/                           # Skrypty do zbierania danych
│   ├── scraper_tauron.py           # Główny skrobacz (PDF → HTML → scoreboard)
│   ├── 01_pobierz_wyniki_setow.py    # Pobieranie punktów setowych
│   ├── 02_scal_wyniki_setow.py       # Scalanie scoreboard z danymi
│   ├── 03_usun_typ_meczu_dodaj_liczbe_setow.py
│   ├── 04_pobierz_daty_meczow.py     # Pobieranie dat meczów
│   └── scal_wyniki_z_danymi.py
├── data/                              # Dane (zobacz .gitignore)
│   ├── processed_postmatch.csv        # Główny zbiór do modelowania
│   └── match_dates.csv
├── requirements.txt                   # Zależności
└── .gitignore
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

1. **Zbieranie danych:** `python scraper/scraper_tauron.py`
2. **Analiza i modelowanie:** otwórz `analiza_i_modelowanie.ipynb` w Jupyter

## Uwagi

- Dane pochodzą ze strony [tauronliga.pl](https://www.tauronliga.pl)
- Analiza ma charakter *ex post* — bada związek statystyk meczowych z wynikiem
- Szczegółowy opis metodologii i wyników znajduje się w pracy licencjackiej
