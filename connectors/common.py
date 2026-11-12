"""Shared helpers for the data connectors: paths, HTTP with retries, CSV upserts."""

from __future__ import annotations

import datetime as dt
import pathlib
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # all timestamps are stored as UTC strings in this format

_session = requests.Session()
_session.headers["User-Agent"] = "derivatives-market-trader data collector (GitHub Actions)"


def get_json(url: str, params: dict | None = None, retries: int = 4, timeout: int = 90):
    """GET a JSON endpoint, retrying on throttling, server errors and network errors."""
    wait = 5
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        time.sleep(wait)
        wait = min(wait * 2, 120)


def to_ts(unix_seconds) -> pd.Series:
    """Unix seconds to UTC timestamp strings."""
    return pd.to_datetime(pd.Series(unix_seconds), unit="s", utc=True).dt.strftime(TS_FORMAT)


def upsert_csv(path: pathlib.Path, new: pd.DataFrame, key: list[str]) -> int:
    """Merge `new` into the CSV at `path` on `key`. New values win; values that are
    missing in `new` keep the stored value. Returns the number of new keys."""
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new = new.drop_duplicates(key, keep="last").set_index(key)
    if path.exists():
        old = pd.read_csv(path, dtype={k: str for k in key}).set_index(key)
        added = len(new.index.difference(old.index))
        merged = new.combine_first(old)
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        merged = merged[cols]
    else:
        added = len(new)
        merged = new
    merged.sort_index().to_csv(path)
    return added


def upsert_partitioned(folder: pathlib.Path, new: pd.DataFrame, key: list[str],
                       ts_col: str = "timestamp_utc") -> int:
    """Upsert into one CSV per UTC month (folder/YYYY-MM.csv), by `ts_col`."""
    if new.empty:
        return 0
    added = 0
    for month, part in new.groupby(new[ts_col].str[:7]):
        added += upsert_csv(folder / f"{month}.csv", part, key)
    return added


def log_run(path: pathlib.Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {"run_utc": dt.datetime.now(dt.timezone.utc).strftime(TS_FORMAT), **run}
    pd.DataFrame([run]).to_csv(path, mode="a", header=not path.exists(), index=False)


def snake(name: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def month_chunks(start: dt.date, end: dt.date, months: int = 3):
    """Yield inclusive (chunk_start, chunk_end) date pairs covering start..end, each
    ending at a calendar-month boundary and spanning at most `months` months."""
    cur = start
    while cur <= end:
        y, m = divmod(cur.month - 1 + months, 12)
        nxt = dt.date(cur.year + y, m + 1, 1)
        yield cur, min(end, nxt - dt.timedelta(days=1))
        cur = nxt
