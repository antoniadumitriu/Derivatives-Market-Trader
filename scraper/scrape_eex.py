"""Daily snapshot of public EEX settlement pages.

For each page in PAGES this script:
  1. loads the page in headless Chromium and accepts the cookie banner,
  2. records every JSON response the page's data widgets fetch (the most
     robust capture, since it does not depend on the page layout),
  3. parses any HTML tables on the rendered page into CSV,
  4. saves a screenshot for debugging.

Output goes to data/YYYY-MM-DD/ (committed by the workflow) and
debug/YYYY-MM-DD/ (uploaded as a workflow artifact, not committed).
The script exits with an error if no page yielded any data, so a broken
run shows up as a failed workflow and GitHub emails you.
"""

import datetime as dt
import json
import pathlib
import re
import sys
from io import StringIO

import pandas as pd
from playwright.sync_api import sync_playwright

# Public EEX market data pages to snapshot. Check these URLs in a browser
# after the first run and adjust if EEX has moved them.
PAGES = {
    "power_futures": "https://www.eex.com/en/market-data/power/futures",
    "power_options": "https://www.eex.com/en/market-data/market-data-hub/power/future-styled-options",
    "environmental_futures": "https://www.eex.com/en/market-data/environmentals/futures",
}

ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = dt.datetime.now(dt.timezone.utc).date().isoformat()
DATA_DIR = ROOT / "data" / TODAY
DEBUG_DIR = ROOT / "debug" / TODAY


def accept_cookies(page) -> None:
    pattern = re.compile(r"accept|akzeptieren|agree", re.I)
    try:
        page.get_by_role("button", name=pattern).first.click(timeout=5_000)
    except Exception:
        pass  # no banner, or already accepted


def scrape_page(page, name: str, url: str) -> dict:
    captured = []

    def on_response(resp):
        try:
            if resp.request.resource_type not in ("xhr", "fetch"):
                return
            if "json" not in resp.headers.get("content-type", ""):
                return
            captured.append({"url": resp.url, "status": resp.status, "body": resp.json()})
        except Exception:
            pass

    page.on("response", on_response)
    result = {"page": name, "url": url, "tables": 0, "json_responses": 0, "error": None}
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        accept_cookies(page)
        try:
            page.wait_for_selector("table", timeout=30_000)
        except Exception:
            pass  # data may only arrive as JSON
        page.wait_for_timeout(8_000)  # let the widgets finish loading

        page.screenshot(path=str(DEBUG_DIR / f"{name}.png"), full_page=True)

        try:
            tables = pd.read_html(StringIO(page.content()))
        except ValueError:
            tables = []
        for i, table in enumerate(tables):
            table.to_csv(DATA_DIR / f"{name}_table{i}.csv", index=False)
        result["tables"] = len(tables)

        if captured:
            with open(DATA_DIR / f"{name}_responses.json", "w") as f:
                json.dump(captured, f, ensure_ascii=False, indent=1)
        result["json_responses"] = len(captured)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        page.remove_listener("response", on_response)
    return result


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(locale="en-GB", viewport={"width": 1600, "height": 2400})
        page = context.new_page()
        for name, url in PAGES.items():
            res = scrape_page(page, name, url)
            print(json.dumps(res))
            results.append(res)
        browser.close()

    manifest = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pages": results}
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    if not any(r["tables"] or r["json_responses"] for r in results):
        print("No data captured from any page. Check the screenshots in the workflow artifact.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
