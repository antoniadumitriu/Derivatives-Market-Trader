# EEX settlement scraper

Collects daily settlement prices for EEX power and carbon futures so that price history builds up over time. The public EEX website only keeps about 45 days of history, so every missed day is lost data. Part of the project "Bayesian Modeling of European Power and Carbon Derivative Markets".

## What it collects

- **Power futures:** base and peak load, month, quarter and year contracts, for every European area EEX lists (DE, FR, AT, NL, BE, IT, ES, CH, CZ, HU, PL, Nordic and the Nordic price areas, and more). Japanese power is excluded.
- **Carbon futures:** EUA, UK allowances (UKA) and ETS2.

About 1,200 contracts in total. The scope is set in `in_scope()` in `scraper/scrape_eex.py`.

## How it works

A GitHub Actions workflow runs every weekday at 19:37 UTC, after EEX publishes settlements. It opens the EEX power futures page in headless Chromium, reads the contract catalog the page downloads from EEX's public market data API, and then requests the last 45 days of settlements for each in-scope contract from within the page, a few at a time with pauses. Each request returns the full 45-day window, so missed days are filled in automatically on the next run.

## Output

- `data/settlements/<year>.csv`: one row per contract, delivery period and trade date, with settlement price, volume and open interest. Rows are upserted, so a corrected settlement replaces the old value.
- `data/contracts.csv`: the in-scope contract catalog as of the latest run.
- `data/runs.csv`: one line per run with counts of contracts, successful and failed requests, and new rows.

A screenshot of the page is uploaded as a workflow artifact for debugging.

## Running manually

On GitHub: Actions tab, "EEX daily settlements", "Run workflow".

Locally:

```
pip install -r requirements.txt
python -m playwright install chromium
python scraper/scrape_eex.py
```

## Notes

- Check the EEX website terms of use regarding automated access. The script spaces its requests out and runs once a day.
- If most requests fail, the run fails on purpose so GitHub emails you. Check the log and the screenshot artifact.
- GitHub may pause scheduled workflows in repositories with no activity for 60 days. The daily commits normally prevent this, but watch for the warning email.
