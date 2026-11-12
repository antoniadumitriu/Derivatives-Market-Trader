"""Daily collection of EEX futures settlements from the public market data API.

How it works:
  1. Open the EEX power futures page in headless Chromium. The page's widgets
     call EEX's public JSON API (api.eex-group.com/pub). We record the request
     headers the page uses and the full contract catalog it downloads.
  2. Filter the catalog to the contracts in SCOPE.
  3. From inside the page (so requests look exactly like the website's own),
     fetch each contract's settlement table for the last LOOKBACK_DAYS days.
  4. Upsert the rows into data/eex/settlements/<year>.csv, keyed by contract,
     delivery period and trade date (latest value wins, so corrections are
     picked up). Write the in-scope catalog to data/eex/contracts.csv and append a
     line to data/eex/runs.csv.

The script exits with an error if most requests fail, so a broken run shows
up as a failed workflow and GitHub emails you.
"""

import base64
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.parse

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = "https://www.eex.com/en/market-data/power/futures"
API = "https://api.eex-group.com/pub"
LOOKBACK_DAYS = 45  # the public site keeps roughly this much history
BATCH = 50          # contracts per in-page batch (progress is logged per batch)
CONCURRENCY = 1     # one request at a time: the API rate-limits bursts
DELAY_MS = 900      # pause between requests, to stay under the rate limit
RETRIES = 4         # retries per request when throttled, with back-off

EXCLUDED_AREAS = {"Chubu", "Kansai", "Tokyo"}  # Japanese power


def in_scope(c: dict) -> bool:
    if c["pricing"] != "F":
        return False
    if c["commodity"] == "POWER":
        return (
            c["product"] in ("Base", "Peak")
            and c["maturityType"] in ("Month", "Quarter", "Year")
            and c["area"] not in EXCLUDED_AREAS
        )
    if c["commodity"] == "ENVIRONMENTALS":
        return c["product"] in ("EUA", "UKA", "ETS2")
    return False


ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "eex"
DEBUG = ROOT / "debug"

KEY = ["short_code", "maturity", "trade_date"]
COLUMNS = [
    "trade_date", "short_code", "commodity", "area", "product", "maturity_type",
    "maturity", "settlement", "volume", "gross_open_interest_lots",
    "gross_open_interest_size", "net_open_interest_lots", "net_open_interest_size",
    "currency", "unit",
]

FETCH_JS = """
async ({urls, headers, concurrency, delay, retries}) => {
  const out = new Array(urls.length);
  let next = 0;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  async function get(url) {
    let wait = 15000;
    for (let attempt = 0; ; attempt++) {
      let res;
      try {
        const r = await fetch(url, {headers});
        res = r.ok ? {status: r.status, body: await r.json()} : {status: r.status};
      } catch (e) {
        res = {status: 0, error: String(e)};  // throttled responses surface as network errors
      }
      const retryable = res.status === 0 || res.status === 429 || res.status >= 500;
      if (!retryable || attempt >= retries) return res;
      await sleep(wait);
      wait = Math.min(wait * 2, 60000);
    }
  }
  async function worker() {
    while (next < urls.length) {
      const i = next++;
      out[i] = await get(urls[i]);
      await sleep(delay);
    }
  }
  await Promise.all(Array.from({length: concurrency}, worker));
  return out;
}
"""


def catalog_url() -> str:
    scope = [{"commodity": "All", "pricing": "All", "area": "All", "product": "All",
              "productSpecific": "All", "maturityType": "All"}]
    data = base64.b64encode(json.dumps(scope, separators=(",", ":")).encode()).decode()
    return f"{API}/customise-widget/filter-data-with-scope?data={data}"


def table_url(c: dict, start: dt.date, end: dt.date) -> str:
    params = {
        "shortCode": c["shortCode"], "commodity": c["commodity"], "pricing": c["pricing"],
        "area": c["area"], "product": c["product"], "maturity": c["maturity"],
        "startDate": start.isoformat(), "endDate": end.isoformat(),
        "maturityType": c["maturityType"], "isRolling": "true",
    }
    return f"{API}/market-data/table-data?" + urllib.parse.urlencode(params)


def rows_from(c: dict, body: dict) -> list[dict]:
    h = body.get("header") or []
    out = []
    for r in body.get("data") or []:
        x = dict(zip(h, r))
        out.append({
            "trade_date": x.get("tradeDate"),
            "short_code": c["shortCode"],
            "commodity": c["commodity"],
            "area": c["area"],
            "product": c["product"],
            "maturity_type": c["maturityType"],
            "maturity": c["maturity"],
            "settlement": x.get("settlPx"),
            "volume": x.get("totVolTrdd"),
            "gross_open_interest_lots": x.get("grossOpenInt"),
            "gross_open_interest_size": x.get("grossOpenIntSz"),
            "net_open_interest_lots": x.get("netOpenInt"),
            "net_open_interest_size": x.get("netOpenIntSz"),
            "currency": body.get("currency"),
            "unit": body.get("uOM"),
        })
    return out


def upsert(rows: list[dict]) -> int:
    """Merge rows into one CSV per trade year. Returns number of new keys."""
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=COLUMNS).dropna(subset=["trade_date"])
    new["maturity"] = new["maturity"].astype(str)
    out_dir = DATA / "settlements"
    out_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for year, part in new.groupby(new["trade_date"].str[:4]):
        path = out_dir / f"{year}.csv"
        if path.exists():
            old = pd.read_csv(path, dtype={"maturity": str})
            before = set(map(tuple, old[KEY].astype(str).values))
            merged = pd.concat([old, part]).drop_duplicates(KEY, keep="last")
        else:
            before = set()
            merged = part.drop_duplicates(KEY, keep="last")
        after = set(map(tuple, merged[KEY].astype(str).values))
        added += len(after - before)
        merged.sort_values(["trade_date", "commodity", "area", "short_code", "maturity"]) \
              .to_csv(path, index=False)
    return added


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(locale="en-GB").new_page()

        api_headers = {}
        catalog = {}

        def on_request(req):
            if req.url.startswith(API) and not api_headers:
                for k, v in req.headers.items():
                    if k.lower() not in ("referer", "origin", "user-agent", "cookie", "host") \
                            and not k.lower().startswith("sec-"):
                        api_headers[k] = v

        def on_response(resp):
            if "filter-data-with-scope" in resp.url and not catalog:
                try:
                    catalog.update(resp.json())
                except Exception:
                    pass

        page.on("request", on_request)
        page.on("response", on_response)
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=60_000)
        try:
            page.get_by_role("button", name="Accept", exact=False).first.click(timeout=5_000)
        except Exception:
            pass
        page.wait_for_timeout(10_000)
        page.screenshot(path=str(DEBUG / "page.png"))

        if not catalog:  # page did not load it; fetch it ourselves
            res = page.evaluate(FETCH_JS, {"urls": [catalog_url()], "headers": api_headers,
                                           "concurrency": 1, "delay": 0, "retries": RETRIES})[0]
            if res.get("status") == 200:
                catalog.update(res["body"])
        if not catalog.get("data"):
            print("Could not load the EEX contract catalog.")
            return 1

        h = catalog["header"]
        everything = pd.DataFrame([dict(zip(h, r)) for r in catalog["data"]])
        # What EEX lists overall, so the scope can be widened knowingly.
        everything.groupby(["commodity", "pricing", "area", "product", "maturityType"], dropna=False) \
            .agg(contracts=("shortCode", "size"), short_codes=("shortCode", lambda x: " ".join(sorted(set(x))))) \
            .reset_index().to_csv(DATA / "catalog_summary.csv", index=False)
        if os.environ.get("EEX_CATALOG_ONLY"):
            print("Catalog summary written; EEX_CATALOG_ONLY is set, so no settlements are fetched.")
            return 0
        contracts = [c for c in (dict(zip(h, r)) for r in catalog["data"]) if in_scope(c)]
        contracts = list({(c["shortCode"], c["maturity"]): c for c in contracts}.values())
        pd.DataFrame(contracts).sort_values(["commodity", "area", "shortCode", "maturity"]) \
            .to_csv(DATA / "contracts.csv", index=False)
        print(f"{len(contracts)} contracts in scope; API headers: {sorted(api_headers)}")

        rows, ok, failed, statuses = [], 0, 0, {}
        for i in range(0, len(contracts), BATCH):
            chunk = contracts[i:i + BATCH]
            results = page.evaluate(FETCH_JS, {
                "urls": [table_url(c, start, today) for c in chunk],
                "headers": api_headers, "concurrency": CONCURRENCY, "delay": DELAY_MS,
                "retries": RETRIES,
            })
            for c, res in zip(chunk, results):
                st = res.get("status", 0)
                statuses[st] = statuses.get(st, 0) + 1
                if st == 200 and isinstance(res.get("body"), dict):
                    ok += 1
                    rows.extend(rows_from(c, res["body"]))
                else:
                    failed += 1
            print(f"  {min(i + BATCH, len(contracts))}/{len(contracts)} done, {failed} failed")
        browser.close()

    added = upsert(rows)
    run = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "contracts": len(contracts), "ok": ok, "failed": failed,
           "rows_fetched": len(rows), "new_rows": added}
    log = DATA / "runs.csv"
    pd.DataFrame([run]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(json.dumps(run), "statuses:", statuses)

    if ok < 0.5 * max(len(contracts), 1):
        print("More than half of the requests failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
