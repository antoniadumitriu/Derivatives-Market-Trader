# Derivatives Market Trader

Data collection, and later modeling, for Bayesian models of European power and carbon derivative markets. Every source used here is free and public and needs no API key or account.

## Layout

```
connectors/
  common.py              shared helpers (HTTP retries, CSV upserts)
  eex/scrape_eex.py      EEX futures settlements (power and carbon)
  energy_charts/fetch.py Energy-Charts: spot prices, generation, forecasts, flows, capacity
  open_meteo/fetch.py    Open-Meteo: weather actuals and forecast snapshots
data/
  eex/                   settlements/<year>.csv, contracts.csv, runs.csv
  energy_charts/         prices/, generation/, forecasts/, crossborder/ (one CSV per UTC month),
                         installed_capacity.csv, runs.csv
  weather/               actuals/, forecasts/ (one CSV per month), points.csv, runs.csv
pipeline/build_db.py     loads all CSVs into data/warehouse.duckdb (local, not committed)
models/de_base_month.py  Bayesian model of the German base month index (NumPyro)
reports/de_base_month/   latest model fit, backtest and comparison with EEX futures
.github/workflows/
  eex-daily.yml          weekdays 19:37 UTC
  fundamentals-daily.yml every day 13:17 UTC (Energy-Charts and Open-Meteo)
```

All timestamps are UTC, written as `YYYY-MM-DDTHH:MM:SSZ`. Every connector upserts: re-running it over a period it has already collected replaces values with the latest published ones and never creates duplicates.

## Datasets

| Folder | Content | Resolution | Units | History |
|---|---|---|---|---|
| `eex/settlements` | ~1,200 futures: power base and peak (month, quarter, year) for all European EEX areas, plus EUA, UKA, ETS2. Settlement price, volume, open interest | daily | EUR/MWh, EUR/t | builds up from Sept 2026 (EEX only shows 45 days) |
| `energy_charts/prices` | Day-ahead auction prices for 16 bidding zones (DE-LU, FR, NL, BE, AT, CH, PL, CZ, HU, SI, IT-North, ES, DK1, DK2, NO2, SE4) | native: hourly to Sept 2025, 15 min after | EUR/MWh | from 2021 |
| `energy_charts/generation` | Net public generation by technology, load, residual load, renewable shares, for DE, FR, NL, BE, AT, CH, PL, CZ | hourly mean | MW (shares in %) | from 2021 |
| `energy_charts/forecasts` | Day-ahead forecasts of solar, onshore and offshore wind, and load, same countries (where published) | hourly mean | MW | from 2021 |
| `energy_charts/crossborder` | Physical cross-border flows by neighbour, same countries. Positive means import into the country | hourly mean | GW | from 2021 |
| `energy_charts/installed_capacity.csv` | Installed capacity by technology (yearly, and monthly where available) | yearly / monthly | GW | full |
| `weather/actuals` | Temperature, 100 m wind speed, solar radiation, cloud cover at 12 points (see `points.csv`) | hourly | °C, m/s, W/m², % | from 2021 |
| `weather/forecasts` | Daily snapshot of the 16-day forecast at the same points, keyed by issue date | hourly | as above | builds up from Nov 2026 |

## Running

The workflows run on their own. To run one by hand: Actions tab, pick the workflow, "Run workflow". The fundamentals workflow takes an optional start date for backfills.

Locally:

```
pip install -r requirements.txt
python -m playwright install chromium     # only needed for EEX
python -m connectors.eex.scrape_eex
python -m connectors.energy_charts.fetch  [--start 2021-01-01] [--datasets prices,generation]
python -m connectors.open_meteo.fetch     [--start 2021-01-01] [--datasets actuals]
```

## Modeling

```
python -m pipeline.build_db                       # build the DuckDB warehouse
python -m models.de_base_month --backtest 24      # fit, forecast the next 3 months, backtest 24 months
python -m models.report_de_base_month             # write reports/de_base_month/README.md and charts
```

The warehouse has raw views over every dataset plus three modeling tables: `de_daily` (one row per German
delivery day with base and peak prices and fundamentals), `de_month_index` (realised monthly base index) and
`de_base_month_futures` (EEX German base month futures by trade date). Results of the latest run are in
[reports/de_base_month](reports/de_base_month/README.md).

## Sources and terms

- **EEX** public market data pages. Check the EEX terms of use regarding automated access; the scraper spaces requests out and runs once a day.
- **Energy-Charts** (Fraunhofer ISE), data under CC BY 4.0, with upstream sources such as ENTSO-E and Bundesnetzagentur/SMARD named in the API responses. Cite them if you publish results.
- **Open-Meteo**, data under CC BY 4.0. The free API is for non-commercial use; a commercial product needs one of their paid plans.

## Notes

- A connector that fails mostly or completely makes its workflow run fail, so GitHub emails you. Data that was collected before the failure is still committed.
- Scheduled GitHub runs often start a few hours late. This does not lose data: every run re-fetches a lookback window (EEX 45 days, Energy-Charts 7 days, weather 14 days).
- GitHub may pause scheduled workflows in repositories with no activity for 60 days. The daily commits normally prevent this.
