"""Energy-Charts (Fraunhofer ISE) connector. Free public API, no key needed.

Datasets (all timestamps UTC, files are one CSV per UTC month):
  prices        day-ahead auction prices per bidding zone, native resolution
                (hourly until Sept 2025, 15-minute after), EUR/MWh
  generation    public net generation by technology plus load and residual load,
                hourly means, MW (share columns are percent)
  forecasts     day-ahead forecasts of solar, wind and load, hourly means, MW
  crossborder   physical cross-border flows per neighbour, hourly means, GW
                (positive = import into the country, as published)
  installed     installed capacity by technology, GW (one file, rewritten each run)

Usage:
  python -m connectors.energy_charts.fetch                  # last 7 days
  python -m connectors.energy_charts.fetch --start 2021-01-01 --datasets prices
Data licence: CC BY 4.0, source Energy-Charts / Fraunhofer ISE (and the
upstream sources named in each response).
"""

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from connectors.common import (DATA, get_json, log_run, month_chunks, snake, to_ts,
                               upsert_csv, upsert_partitioned)

API = "https://api.energy-charts.info"
OUT = DATA / "energy_charts"

ZONES = ["DE-LU", "FR", "NL", "BE", "AT", "CH", "PL", "CZ", "HU", "SI",
         "IT-North", "ES", "DK1", "DK2", "NO2", "SE4"]
COUNTRIES = ["de", "fr", "nl", "be", "at", "ch", "pl", "cz"]
FORECAST_TYPES = ["solar", "wind_onshore", "wind_offshore", "load"]
LOOKBACK_DAYS = 7
PAUSE = 1.0  # seconds between requests (the API returns 429 on bursts)


class Stats:
    def __init__(self):
        self.ok = self.failed = self.new_rows = 0
        self.errors: list[str] = []

    def fail(self, what: str, err: Exception):
        self.failed += 1
        self.errors.append(f"{what}: {err}")
        print(f"  FAILED {what}: {err}")


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average a wide frame indexed by UTC datetime to hourly, dropping empty hours."""
    out = df.resample("h").mean().dropna(how="all")
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "timestamp_utc"
    return out.reset_index()


def frame(unix_seconds, columns: dict) -> pd.DataFrame:
    idx = pd.to_datetime(pd.Series(unix_seconds), unit="s", utc=True)
    df = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce").values
                       for k, v in columns.items()}, index=idx.values)
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def prices(start, end, s: Stats):
    for zone in ZONES:
        for a, b in month_chunks(start, end):
            try:
                d = get_json(f"{API}/price", {"bzn": zone, "start": a.isoformat(), "end": b.isoformat()})
                if d.get("price"):
                    df = pd.DataFrame({"timestamp_utc": to_ts(d["unix_seconds"]), "zone": zone,
                                       "price_eur_mwh": pd.to_numeric(pd.Series(d["price"]), errors="coerce")})
                    s.new_rows += upsert_partitioned(OUT / "prices", df.dropna(subset=["price_eur_mwh"]),
                                                     ["timestamp_utc", "zone"])
                s.ok += 1
            except Exception as e:
                s.fail(f"price {zone} {a}", e)
            time.sleep(PAUSE)
        print(f"  prices {zone} done")


def generation(start, end, s: Stats):
    for c in COUNTRIES:
        for a, b in month_chunks(start, end):
            try:
                d = get_json(f"{API}/public_power", {"country": c, "start": a.isoformat(), "end": b.isoformat()})
                if d.get("unix_seconds"):
                    df = hourly(frame(d["unix_seconds"],
                                      {snake(p["name"]): p["data"] for p in d["production_types"]}))
                    df.insert(1, "country", c)
                    s.new_rows += upsert_partitioned(OUT / "generation", df, ["timestamp_utc", "country"])
                s.ok += 1
            except Exception as e:
                s.fail(f"public_power {c} {a}", e)
            time.sleep(PAUSE)
        print(f"  generation {c} done")


def forecasts(start, end, s: Stats):
    for c in COUNTRIES:
        for a, b in month_chunks(start, end):
            cols, idx = {}, None
            for t in FORECAST_TYPES:
                try:
                    d = get_json(f"{API}/public_power_forecast",
                                 {"country": c, "production_type": t, "forecast_type": "day-ahead",
                                  "start": a.isoformat(), "end": b.isoformat()})
                    if d.get("unix_seconds"):
                        part = frame(d["unix_seconds"], {f"{t}_day_ahead": d["forecast_values"]})
                        idx = part if idx is None else idx.join(part, how="outer")
                    s.ok += 1
                except Exception as e:
                    # Not every country publishes every forecast type; the API answers
                    # 400 or 404 for those, which is expected.
                    if not any(code in str(e) for code in ("400 Client", "404 Client")):
                        s.fail(f"forecast {c} {t} {a}", e)
                time.sleep(PAUSE)
            if idx is not None:
                df = hourly(idx)
                df.insert(1, "country", c)
                s.new_rows += upsert_partitioned(OUT / "forecasts", df, ["timestamp_utc", "country"])
        print(f"  forecasts {c} done")


def crossborder(start, end, s: Stats):
    for c in COUNTRIES:
        for a, b in month_chunks(start, end):
            try:
                d = get_json(f"{API}/cbpf", {"country": c, "start": a.isoformat(), "end": b.isoformat()})
                if d.get("unix_seconds"):
                    df = hourly(frame(d["unix_seconds"], {snake(p["name"]): p["data"] for p in d["countries"]}))
                    df.insert(1, "country", c)
                    s.new_rows += upsert_partitioned(OUT / "crossborder", df, ["timestamp_utc", "country"])
                s.ok += 1
            except Exception as e:
                s.fail(f"cbpf {c} {a}", e)
            time.sleep(PAUSE)
        print(f"  crossborder {c} done")


def installed(start, end, s: Stats):
    rows = []
    for c in COUNTRIES:
        for step in ("yearly", "monthly"):
            try:
                d = get_json(f"{API}/installed_power", {"country": c, "time_step": step}, retries=1)
                for p in d.get("production_types", []):
                    for period, v in zip(d["time"], p["data"]):
                        if v is None:
                            continue
                        if step == "monthly":  # "MM.YYYY" -> "YYYY-MM"
                            period = f"{period[3:]}-{period[:2]}"
                        rows.append({"country": c, "time_step": step, "period": period,
                                     "technology": snake(p["name"]), "capacity_gw": v})
                s.ok += 1
            except Exception as e:
                if not any(code in str(e) for code in ("400 Client", "404 Client")):
                    # monthly data only exists for some countries
                    s.fail(f"installed {c} {step}", e)
            time.sleep(PAUSE)
    if rows:
        upsert_csv(OUT / "installed_capacity.csv", pd.DataFrame(rows),
                   ["country", "time_step", "period", "technology"])
    print(f"  installed capacity: {len(rows)} values")


DATASETS = {"prices": prices, "generation": generation, "forecasts": forecasts,
            "crossborder": crossborder, "installed": installed}


def main() -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=LOOKBACK_DAYS))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=today + dt.timedelta(days=1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    s = Stats()
    for name in args.datasets.split(","):
        print(f"{name}: {args.start} to {args.end}")
        DATASETS[name](args.start, args.end, s)

    log_run(OUT / "runs.csv", {"start": args.start, "end": args.end, "datasets": args.datasets,
                               "ok": s.ok, "failed": s.failed, "new_rows": s.new_rows})
    print(f"ok={s.ok} failed={s.failed} new_rows={s.new_rows}")
    if s.failed > 0.2 * max(s.ok + s.failed, 1):
        print("More than 20% of requests failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
