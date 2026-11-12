"""Write the model report (README.md and charts) from the model outputs.

  python -m models.report_de_base_month             # v2 (with gas and carbon), compared with v1
  python -m models.report_de_base_month --model v1
"""

import argparse

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.de_base_month import OUT as OUT_V1
from models.de_base_month_v2 import OUT as OUT_V2
from pipeline.build_db import DB

MODEL = "#2a78d6"     # categorical slot 1
OBSERVED = "#eb6834"  # categorical slot 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)


def label(month: str) -> str:
    return pd.Timestamp(f"{month[:4]}-{month[4:]}-01").strftime("%b %y")


def intervals(ax, x, df, observed, observed_label):
    ax.vlines(x, df["p05"], df["p95"], color=MODEL, linewidth=2, alpha=0.35, label="Model 90% interval")
    ax.vlines(x, df["p25"], df["p75"], color=MODEL, linewidth=6, label="Model 50% interval")
    ax.scatter(x, df["p50"], s=36, color=SURFACE, edgecolor=MODEL, linewidth=2, zorder=3, label="Model median")
    ax.scatter(x, observed, s=64, color=OBSERVED, edgecolor=SURFACE, linewidth=2, zorder=4, label=observed_label)


DESCRIPTION = {
    "v1": """It models the daily German day-ahead base price as a weekly-varying **price level**, plus a weekly-varying
**merit-order slope** times standardised residual load (load minus wind and solar), plus a weekend effect,
with heavy-tailed (Student-t) noise whose scale also varies week to week. All three weekly states are random walks.
A delivery month is forecast by carrying the states forward along their random walks and drawing residual load
from the same calendar month of the last 3 years, then averaging simulated daily prices into the month index
that a German base month future settles against.

It does not see gas or carbon prices. The weekly level stands in for them, so the model knows
"prices are high right now" but not "the gas curve says they will fall next quarter".""",
    "v2": """Version 2 ties the price to the **cost of gas-fired power**: TTF gas plus the carbon cost of burning it,
divided by a 50% plant efficiency. The daily price is that cost times a weekly-varying **multiple**, plus a
weekly-varying slope on standardised residual load (also scaled by the cost), plus a weekend effect, with
heavy-tailed noise that partly scales with the cost. The multiple and slope are random walks, so the model still
adapts when coal, renewables or imports take over price setting, but a move in gas now moves its forecast
straight away.

A delivery month is forecast from the **TTF futures price for that month** and the EUA December future, with
gas-price uncertainty that grows with time to delivery (volatility from the last 60 days of TTF), and residual
load drawn from the same calendar month of the last 3 years. In the backtest, where old TTF contract prices are
not available, the TTF front month at the forecast date stands in for the delivery month's gas price.""",
}
STATE_LABELS = {
    "v1": {"tau_level": "Weekly change in level (s.d.), EUR/MWh", "tau_beta": "Weekly change in slope (s.d.)",
           "level_last_week": "Current level, EUR/MWh", "beta_last_week": "Current slope, EUR/MWh per s.d. of residual load",
           "sigma_last_week": "Current daily noise scale, EUR/MWh", "gamma": "Weekend effect, EUR/MWh",
           "nu": "Student-t degrees of freedom"},
    "v2": {"tau_m": "Weekly change in price/cost multiple (s.d.)", "tau_s": "Weekly change in slope (s.d.)",
           "m_last_week": "Current price/cost multiple", "s_last_week": "Current slope, multiple per s.d. of residual load",
           "phi": "Share of volatility that scales with fuel cost (0 to 1)", "gamma": "Weekend effect, EUR/MWh",
           "nu": "Student-t degrees of freedom"},
}


def metrics(bt):
    return {"mae": bt.abs_err_mean.mean(), "naive": bt.abs_err_naive.mean(), "in50": bt.in_50.mean(),
            "in90": bt.in_90.mean(), "crps": bt.crps.mean(), "bias": (bt["mean"] - bt.actual).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2"], default="v2")
    version = ap.parse_args().model
    OUT = OUT_V2 if version == "v2" else OUT_V1
    other = OUT_V1 if version == "v2" else None
    bt = pd.read_csv(OUT / "backtest.csv", dtype={"target": str}).set_index("target")
    fc = pd.read_csv(OUT / "forecast.csv", dtype={"month": str}).set_index("month")
    params = json.loads((OUT / "params.json").read_text())
    con = duckdb.connect(str(DB), read_only=True)
    fut = con.execute("SELECT trade_date, month, settlement FROM de_base_month_futures").df()
    fut["trade_date"] = pd.to_datetime(fut["trade_date"])

    # Futures at each backtest origin, where the scraper already has them.
    bt["futures_at_origin"] = [
        fut[(fut["month"] == t) & (fut["trade_date"] <= pd.Timestamp(o))].sort_values("trade_date")["settlement"].iloc[-1]
        if len(fut[(fut["month"] == t) & (fut["trade_date"] <= pd.Timestamp(o))]) else None
        for t, o in zip(bt.index, bt["origin"])]

    # Chart 1: backtest
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = range(len(bt))
    intervals(ax, x, bt, bt["actual"], "Realised base index")
    ax.set_xticks(list(x), [label(m) for m in bt.index], rotation=0, fontsize=8)
    style(ax, "One-month-ahead forecasts of the German base month index vs. outcome", "EUR/MWh")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=4, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "backtest.png", dpi=150)

    # Chart 2: forward months vs EEX futures
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(fc))
    intervals(ax, x, fc, fc["futures"], f"EEX settlement {fc['futures_trade_date'].iloc[0]}")
    ax.set_xticks(list(x), [label(m) for m in fc.index])
    for i, (m, r) in enumerate(fc.iterrows()):
        ax.annotate(f"{r['futures']:.0f}", (i, r["futures"]), xytext=(10, -3), textcoords="offset points",
                    color=INK, fontsize=9)
    style(ax, f"German base months: model vs. futures (data to {params['origin']})", "EUR/MWh")
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_vs_futures.png", dpi=150)

    n = len(bt)
    p = params["params"]
    fmt = lambda v: "" if pd.isna(v) else f"{v:.1f}"
    rows_bt = "\n".join(
        f"| {label(t)} | {r.actual:.1f} | {r['mean']:.1f} | {r.p05:.1f} to {r.p95:.1f} | {r.naive_30d:.1f} | "
        f"{fmt(r.futures_at_origin)} | {r.pit:.2f} |" for t, r in bt.iterrows())
    rows_fc = "\n".join(
        f"| {label(m)} | {r['mean']:.1f} | {r.p25:.1f} to {r.p75:.1f} | {r.p05:.1f} to {r.p95:.1f} | {r.futures:.1f} | "
        f"{r.p_futures_above_model:.0%} |" for m, r in fc.iterrows())

    param_rows = "\n".join(f"| {label} | {p[k]['mean']:.2f} | {p[k]['p05']:.2f} to {p[k]['p95']:.2f} |"
                           for k, label in STATE_LABELS[version].items() if k in p)
    compare = ""
    if other is not None and (other / "backtest.csv").exists():
        a, b = metrics(pd.read_csv(other / "backtest.csv")), metrics(bt)
        compare = f"""
### Compared with version 1 (same months)

| | Version 1 | Version 2 |
|---|---|---|
| Mean absolute error, EUR/MWh | {a['mae']:.1f} | {b['mae']:.1f} |
| Mean error (bias), EUR/MWh | {a['bias']:+.1f} | {b['bias']:+.1f} |
| 50% interval coverage | {a['in50']:.0%} | {b['in50']:.0%} |
| 90% interval coverage | {a['in90']:.0%} | {b['in90']:.0%} |
| Mean CRPS (lower is better) | {a['crps']:.1f} | {b['crps']:.1f} |
| Naive forecast error, EUR/MWh | {a['naive']:.1f} | {b['naive']:.1f} |
"""
    m = metrics(bt)
    gain = 1 - m["mae"] / m["naive"]
    cover = ("Its intervals are too wide rather than too narrow (90% intervals caught "
             f"{m['in90']:.0%} of outcomes), the safer failure." if m["in90"] > 0.95 else
             f"Its 90% intervals caught {m['in90']:.0%} of outcomes and its 50% intervals {m['in50']:.0%}, "
             "close to the targets." if m["in90"] >= 0.8 else
             f"Its intervals are too narrow: 90% intervals caught only {m['in90']:.0%} of outcomes.")
    lean = (f"It leans {'high' if m['bias'] > 0 else 'low'} on average by {abs(m['bias']):.1f} EUR/MWh"
            + (", mostly in months where prices fell, because it has no fuel prices to tell it a decline is coming."
               if version == "v1" else "."))
    month_of = bt.index.str[4:].astype(int)
    err = bt["mean"] - bt["actual"]
    summer, rest = err[(month_of >= 6) & (month_of <= 10)].mean(), err[(month_of < 6) | (month_of > 10)].mean()
    season = (f" The lean is seasonal: {summer:+.1f} EUR/MWh from June to October against {rest:+.1f} in the other "
              "months. In summer, solar pushes gas out of the price-setting position for many hours, and the model "
              "has no seasonal term to expect that." if summer - rest > 5 else "")
    findings = (f"The model {'beats' if gain > 0 else 'trails'} the naive forecast by {abs(gain):.0%} on absolute error. "
                f"{cover} {lean}{season}")
    md = f"""# German base month model, version {version[1]}

Generated by `python -m models.{"de_base_month_v2" if version == "v2" else "de_base_month"} --backtest {n}` and `python -m models.report_de_base_month --model {version}`.
Data through {params['origin']}; EEX futures as of {fc['futures_trade_date'].iloc[0]}.

## What the model does

{DESCRIPTION[version]}

## Fit

{params['weeks']} weeks of data, {params['seconds']} s of sampling, {params['divergences']} divergences.

| Parameter | Posterior mean | 90% interval |
|---|---|---|
{param_rows}

## Backtest: {n} one-month-ahead forecasts

Refit at each month end, forecast the next month's base index.

| | Model | Naive (last 30 days) |
|---|---|---|
| Mean absolute error | {bt.abs_err_mean.mean():.1f} EUR/MWh | {bt.abs_err_naive.mean():.1f} EUR/MWh |
| 50% interval coverage | {bt.in_50.mean():.0%} (target 50%) | |
| 90% interval coverage | {bt.in_90.mean():.0%} (target 90%) | |
| Mean CRPS | {bt.crps.mean():.1f} | |

{compare}
![Backtest](backtest.png)

| Month | Realised | Model mean | Model 90% | Naive | EEX futures at origin | PIT |
|---|---|---|---|---|---|---|
{rows_bt}

**What this says.** {findings}

PIT is the share of model draws below the outcome; for a calibrated model these are spread evenly between 0 and 1.
Futures at origin are only available from late September 2026, when the scraper started.

## Forward months vs. EEX futures

![Forecast vs futures](forecast_vs_futures.png)

| Month | Model mean | Model 50% | Model 90% | EEX settlement | Share of model draws below futures |
|---|---|---|---|---|---|
{rows_fc}

The last column is the model's probability that the month settles below the current futures price.
Around 50% means model and market agree; well above 50% means the model thinks futures are rich.

## Next improvements

1. A seasonal term (by calendar month) on the price/cost multiple, for the summer over-forecast.
2. Use the futures curve itself as an input once enough history builds up, and model the risk premium
   (futures minus realised) directly.
3. Trend in residual load: recent years are sampled with equal weight, but renewable capacity keeps growing.
4. Coal and imports: when gas is not the marginal plant, the price/cost multiple has to absorb it.
"""
    (OUT / "README.md").write_text(md)
    print(md[:400])


if __name__ == "__main__":
    main()
