# Derivatives Market Trader

Data collection, Bayesian modeling and a dashboard for European power and carbon derivative markets. Every source used here is free and public and needs no API key or account.

## Dashboard

```
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Pages:

| Page | What it shows |
|---|---|
| Overview | Headline prices (spot, front month, EUA, model) and data health: latest data point per dataset and the status of the last collection run per source |
| EEX futures | Forward curves for any power area (base or peak), gas hub (TTF, THE) or carbon product on any trade date, compared with an earlier date, plus settlement history per contract |
| Day-ahead prices | Daily, weekly or monthly prices for up to 8 bidding zones, the average daily shape, and the share of negative-price intervals |
| Gas and carbon | TTF gas, EUA carbon, the cost of gas-fired power against the German power price, and the TTF forward curve |
| Fundamentals | Generation mix and load, price vs. residual load (the merit order), wind and solar forecast vs. actual, cross-border flows and installed capacity, per country |
| Weather | Actuals and the latest 16-day forecast at the tracked points, and German temperature vs. normal |
| German base month | The Bayesian model (version 3, or earlier versions for comparison): forecasts vs. EEX futures, the 24-month backtest, and the weekly states it estimates. A sidebar button refits it on the latest data |

Every chart has hover values and a table view. The dashboard reads `data/warehouse.duckdb`, which it rebuilds from the CSVs on start and whenever they change (for example after `git pull`).

## Layout

```
connectors/
  common.py              shared helpers (HTTP retries, CSV upserts)
  eex/scrape_eex.py      EEX futures settlements (power and carbon)
  energy_charts/fetch.py Energy-Charts: spot prices, generation, forecasts, flows, capacity
  open_meteo/fetch.py    Open-Meteo: weather actuals and forecast snapshots
  yahoo/fetch.py         Yahoo Finance: TTF gas, carbon proxy, EUR/USD
data/
  eex/                   settlements/<year>.csv, contracts.csv, runs.csv
  energy_charts/         prices/, generation/, forecasts/, crossborder/ (one CSV per UTC month),
                         installed_capacity.csv, runs.csv
  weather/               actuals/, forecasts/ (one CSV per month), points.csv, runs.csv
dashboard/                Streamlit dashboard (app.py, views/, lib.py)
pipeline/build_db.py     loads all CSVs into data/warehouse.duckdb (local, not committed)
models/de_base_month_v3.py  current model: v2 with today's solar fleet in the residual-load scenarios
                            and a calendar-month term (each fix also available alone as a variant)
models/de_base_month_v2.py  Bayesian model anchored on gas and carbon costs (the code v3 builds on)
models/de_base_month.py     version 1, price level only (kept for comparison)
reports/de_base_month_v3_both/   latest fit, backtest, comparison with all versions and with EEX futures
reports/de_base_month_*/         the same for the other versions
.github/workflows/
  eex-daily.yml          weekdays 19:37 UTC
  fundamentals-daily.yml every day 13:17 UTC (Energy-Charts, Open-Meteo, Yahoo Finance)
```

All timestamps are UTC, written as `YYYY-MM-DDTHH:MM:SSZ`. Every connector upserts: re-running it over a period it has already collected replaces values with the latest published ones and never creates duplicates.

## Datasets

| Folder | Content | Resolution | Units | History |
|---|---|---|---|---|
| `eex/settlements` | ~1,450 futures: power base and peak (month, quarter, year) for all European EEX areas; EUA, UKA, ETS2; TTF and THE gas (month, quarter, season, year). Settlement price, volume, open interest | daily | EUR/MWh, EUR/t | builds up from Sept 2026 (EEX only shows 45 days); gas from Nov 2026 |
| `eex/catalog_summary.csv` | Everything EEX lists, grouped by commodity, area, product and maturity type, to decide what to add | per run | | |
| `market/daily.csv` | TTF gas front month (Yahoo Finance), WisdomTree Carbon ETC as an EUA proxy, EUR/USD | daily | EUR/MWh, USD, rate | from 2020 |
| `market/ttf_curve.csv` | TTF monthly contracts by trade date (the gas forward curve) | daily | EUR/MWh | from listing of each contract |
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
python -m models.de_base_month_v3 --backtest 24     # current model: fit, forecast 3 months, backtest 24 (~50 min)
python -m models.report_de_base_month               # write the report and charts
# other versions: models.de_base_month_v2, models.de_base_month, models.de_base_month_v3 --variant solar|seasonal
```

The warehouse has raw views over every dataset plus three modeling tables: `de_daily` (one row per German
delivery day with base and peak prices and fundamentals), `de_month_index` (realised monthly base index) and
`de_base_month_futures` (EEX German base month futures by trade date), plus `fuel_daily` (TTF gas, EUA carbon and
the cost of gas-fired power). Results of the latest run are in [reports/de_base_month_v3_both](reports/de_base_month_v3_both/README.md).

| 24-month backtest | v1 | v2 | v2 + today's solar | v2 + calendar month | **v3 (both)** |
|---|---|---|---|---|---|
| Mean absolute error, EUR/MWh | 11.6 | 10.7 | 9.2 | 9.7 | **9.1** |
| Bias, EUR/MWh | +4.5 | +4.6 | -2.7 | +4.6 | -2.6 |
| Bias June to October | +12.7 | +12.6 | +2.2 | +9.1 | -1.3 |
| 50% / 90% interval coverage | 58% / 100% | 54% / 88% | 50% / 88% | 58% / 96% | 58% / 92% |
| CRPS | 8.3 | 7.8 | 6.8 | 7.2 | **6.6** |

## Sources and terms

- **EEX** public market data pages. Check the EEX terms of use regarding automated access; the scraper spaces requests out and runs once a day.
- **Energy-Charts** (Fraunhofer ISE), data under CC BY 4.0, with upstream sources such as ENTSO-E and Bundesnetzagentur/SMARD named in the API responses. Cite them if you publish results.
- **Yahoo Finance** chart data, for personal and non-commercial use. A commercial product should switch to EEX or ICE data.
- **Open-Meteo**, data under CC BY 4.0. The free API is for non-commercial use; a commercial product needs one of their paid plans.

## Notes

- A connector that fails mostly or completely makes its workflow run fail, so GitHub emails you. Data that was collected before the failure is still committed.
- Scheduled GitHub runs often start a few hours late. This does not lose data: every run re-fetches a lookback window (EEX 45 days, Energy-Charts 7 days, weather 14 days).
- GitHub may pause scheduled workflows in repositories with no activity for 60 days. The daily commits normally prevent this.
