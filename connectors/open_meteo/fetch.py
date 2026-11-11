"""Open-Meteo weather connector. Free API, no key needed.

Datasets (hourly, UTC):
  actuals     reanalysis-based history (ERA5 and ECMWF IFS) from the archive API,
              one CSV per UTC month in data/weather/actuals/
  forecasts   a daily snapshot of the 16-day forecast, so forecast history builds
              up over time (Open-Meteo does not keep old forecasts for free);
              one CSV per month of issue date in data/weather/forecasts/

Variables: 2 m temperature (deg C), 100 m wind speed (m/s), shortwave radiation
(W/m2) and cloud cover (%), at the points in POINTS, chosen for demand (large
cities), wind (north German coast and offshore) and solar (south Germany).

Usage:
  python -m connectors.open_meteo.fetch                       # actuals for the last 14 days + today's forecast
  python -m connectors.open_meteo.fetch --start 2021-01-01 --datasets actuals
Note: the free Open-Meteo API is for non-commercial use (CC BY 4.0 data).
A commercial project needs one of their paid plans.
"""

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from connectors.common import DATA, get_json, log_run, upsert_csv

OUT = DATA / "weather"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
LOOKBACK_DAYS = 14

# point id, country, latitude, longitude, why it is included
POINTS = [
    ("de_hamburg", "de", 53.55, 9.99, "demand, north"),
    ("de_berlin", "de", 52.52, 13.40, "demand, east"),
    ("de_ruhr", "de", 51.46, 7.01, "demand, industry"),
    ("de_frankfurt", "de", 50.11, 8.68, "demand, centre"),
    ("de_munich", "de", 48.14, 11.58, "demand and solar, south"),
    ("de_stuttgart", "de", 48.78, 9.18, "solar, south-west"),
    ("de_nordfriesland", "de", 54.60, 8.90, "onshore wind, north coast"),
    ("de_brandenburg", "de", 52.80, 12.50, "onshore wind and solar, east"),
    ("de_north_sea", "de", 54.30, 6.50, "offshore wind, North Sea"),
    ("de_baltic", "de", 54.70, 13.80, "offshore wind, Baltic"),
    ("fr_paris", "fr", 48.86, 2.35, "French demand (heating-sensitive)"),
    ("nl_amsterdam", "nl", 52.37, 4.90, "Dutch demand and wind"),
]


def params(**extra) -> dict:
    return {"latitude": ",".join(str(p[2]) for p in POINTS),
            "longitude": ",".join(str(p[3]) for p in POINTS),
            "hourly": ",".join(VARIABLES), "timezone": "UTC", "wind_speed_unit": "ms", **extra}


def to_frame(body) -> pd.DataFrame:
    locs = body if isinstance(body, list) else [body]
    frames = []
    for point, loc in zip(POINTS, locs):
        h = loc["hourly"]
        df = pd.DataFrame({v: h[v] for v in VARIABLES})
        df.insert(0, "point", point[0])
        df.insert(0, "timestamp_utc", pd.to_datetime(h["time"]).strftime("%Y-%m-%dT%H:%M:%SZ"))
        frames.append(df)
    return pd.concat(frames).dropna(subset=VARIABLES, how="all")


def actuals(start: dt.date, end: dt.date) -> int:
    added, cur = 0, start
    while cur <= end:
        chunk_end = min(end, dt.date(cur.year, 12, 31))
        df = to_frame(get_json(ARCHIVE, params(start_date=cur.isoformat(), end_date=chunk_end.isoformat())))
        for month, part in df.groupby(df["timestamp_utc"].str[:7]):
            added += upsert_csv(OUT / "actuals" / f"{month}.csv", part, ["timestamp_utc", "point"])
        print(f"  actuals {cur} to {chunk_end}: {len(df)} rows")
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(5)
    return added


def forecasts(issue: dt.date) -> int:
    df = to_frame(get_json(FORECAST, params(forecast_days=16)))
    df.insert(0, "issue_date", issue.isoformat())
    print(f"  forecast issued {issue}: {len(df)} rows")
    return upsert_csv(OUT / "forecasts" / f"{issue:%Y-%m}.csv", df, ["issue_date", "timestamp_utc", "point"])


def main() -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=LOOKBACK_DAYS))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=today)
    ap.add_argument("--datasets", default="actuals,forecasts")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(POINTS, columns=["point", "country", "latitude", "longitude", "purpose"]) \
        .to_csv(OUT / "points.csv", index=False)

    run, failed = {"start": args.start, "end": args.end, "datasets": args.datasets}, []
    if "actuals" in args.datasets:
        try:
            run["actuals_new_rows"] = actuals(args.start, args.end)
        except Exception as e:
            failed.append(f"actuals: {e}")
    if "forecasts" in args.datasets:
        try:
            run["forecast_rows"] = forecasts(today)
        except Exception as e:
            failed.append(f"forecasts: {e}")
    run["errors"] = " | ".join(failed)
    log_run(OUT / "runs.csv", run)
    print(run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
