# EEX settlement scraper

Takes a daily snapshot of public EEX settlement pages (power futures, power options, environmental futures) so that price history builds up over time. The public site only shows the last 45 days, so every missed day is lost history.

## How it works

A GitHub Actions workflow runs every weekday at 19:37 UTC. It loads each page in headless Chromium, saves every JSON response the page's data widgets fetch, parses any HTML tables to CSV, and commits the results to `data/YYYY-MM-DD/`. Screenshots are uploaded as a workflow artifact for debugging.

## Output

For each page, a day's folder can contain:

- `<page>_responses.json`: raw JSON fetched by the page (most complete, layout-independent)
- `<page>_tableN.csv`: tables parsed from the rendered page
- `manifest.json`: what was captured from each page and any errors

Parsing these into a clean settlements table is a separate step.

## Running manually

On GitHub: Actions tab, "EEX daily settlements", "Run workflow".

Locally:

```
pip install -r requirements.txt
python -m playwright install chromium
python scraper/scrape_eex.py
```

## Notes

- Check the EEX website terms of use regarding automated access. Keep the request rate low (this script loads three pages once a day).
- If EEX changes its page layout or URLs, update `PAGES` in `scraper/scrape_eex.py`. A run that captures nothing fails on purpose, so GitHub emails you.
- GitHub may pause scheduled workflows in repositories with no activity for 60 days. Watch for the warning email and re-enable from the Actions tab if needed.
