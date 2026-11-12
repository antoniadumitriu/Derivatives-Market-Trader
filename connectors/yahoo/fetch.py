"""Gas and carbon prices from Yahoo Finance's public chart endpoint. No key needed.

Collected (daily closes):
  TTF=F      Dutch TTF natural gas, front-month future (NYMEX-listed), EUR/MWh
  TTF<M><YY>.NYM   individual TTF monthly contracts for the next 18 delivery months,
             stored as a curve: data/market/ttf_curve.csv (trade date x delivery month)
  CARB.L     WisdomTree Carbon ETC, which tracks ICE EUA futures, USD. Used only as a
             proxy for EUA history before the EEX scraper started (Sept 2026)
  EURUSD=X   EUR/USD, to convert the carbon proxy to euros

Files: data/market/daily.csv (symbol, date, close), data/market/ttf_curve.csv,
data/market/runs.csv.

Usage:
  python -m connectors.yahoo.fetch                  # last 30 days
  python -m connectors.yahoo.fetch --start 2020-01-01
Note: Yahoo Finance data is for personal, non-commercial use.
"""

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from connectors.common import DATA, get_json, log_run, upsert_csv

OUT = DATA / "market"
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
SERIES = ["TTF=F", "CARB.L", "EURUSD=X"]
MONTH_CODES = "FGHJKMNQUVXZ"
CURVE_MONTHS = 18
LOOKBACK_DAYS = 30


def chart(symbol: str, start: dt.date) -> pd.DataFrame:
    p1 = int(dt.datetime.combine(start, dt.time(), dt.timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    body = get_json(CHART.format(symbol), {"period1": p1, "period2": p2, "interval": "1d"})
    res = (body.get("chart") or {}).get("result")
    if not res or not res[0].get("timestamp"):
        return pd.DataFrame(columns=["date", "close"])
    r = res[0]
    # Yahoo stamps daily bars at the exchange's session start; convert to the exchange's local date.
    tz = r["meta"].get("exchangeTimezoneName", "UTC")
    dates = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).date
    df = pd.DataFrame({"date": [d.isoformat() for d in dates], "close": r["indicators"]["quote"][0]["close"]})
    return df.dropna().drop_duplicates("date", keep="last")


def contracts(today: dt.date) -> list[tuple[str, str]]:
    """(symbol, delivery month YYYYMM) for the next CURVE_MONTHS delivery months."""
    out = []
    y, m = today.year, today.month
    for _ in range(CURVE_MONTHS):
        m += 1
        if m > 12:
            y, m = y + 1, 1
        out.append((f"TTF{MONTH_CODES[m - 1]}{y % 100:02d}.NYM", f"{y}{m:02d}"))
    return out


def main() -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=LOOKBACK_DAYS))
    args = ap.parse_args()

    ok, failed, errors, new = 0, 0, [], 0
    frames = []
    for s in SERIES:
        try:
            df = chart(s, args.start)
            df.insert(0, "symbol", s)
            frames.append(df)
            ok += 1
            print(f"  {s}: {len(df)} days")
        except Exception as e:
            failed += 1
            errors.append(f"{s}: {e}")
            print(f"  FAILED {s}: {e}")
        time.sleep(1)
    if frames:
        new += upsert_csv(OUT / "daily.csv", pd.concat(frames), ["symbol", "date"])

    curve = []
    for sym, month in contracts(today):
        try:
            df = chart(sym, args.start)
            if not df.empty:
                df = df.rename(columns={"date": "trade_date"})
                df.insert(1, "delivery_month", month)
                curve.append(df)
            ok += 1
        except Exception as e:
            # Far contracts are sometimes not listed on Yahoo yet; that is not a failure.
            if "404" not in str(e):
                failed += 1
                errors.append(f"{sym}: {e}")
                print(f"  FAILED {sym}: {e}")
        time.sleep(1)
    if curve:
        c = pd.concat(curve)
        new += upsert_csv(OUT / "ttf_curve.csv", c, ["trade_date", "delivery_month"])
        print(f"  TTF curve: {c['delivery_month'].nunique()} contracts, {len(c)} rows")

    log_run(OUT / "runs.csv", {"start": args.start, "ok": ok, "failed": failed, "new_rows": new,
                               "errors": " | ".join(errors)[:500]})
    print(f"ok={ok} failed={failed} new_rows={new}")
    return 1 if failed > 0.2 * max(ok + failed, 1) else 0


if __name__ == "__main__":
    sys.exit(main())
