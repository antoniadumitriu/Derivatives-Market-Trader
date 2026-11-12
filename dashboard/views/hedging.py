"""Hedging summary: where the contracts a buyer hedges with (months, quarters, calendar years)
stand today and how they moved over the last day, week and four weeks, with gas and carbon
alongside because they drive the power price."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import SERIES, contract_label, fig_defaults, page, q, rebuild_button, show

page("Hedging summary",
     "The contracts used to hedge power: the next months, quarters and calendar years, with their moves "
     "over the last day, week and four weeks. EEX settlements; history starts in Sept 2026.")
rebuild_button()

f = st.columns([1, 1, 3])
areas = q("SELECT DISTINCT area FROM eex_settlements WHERE commodity = 'POWER' ORDER BY area")["area"].tolist()
area = f[0].selectbox("Power area", areas, index=areas.index("DE") if "DE" in areas else 0)
load = f[1].selectbox("Load", ["Base", "Peak"])

POWER = ("commodity = 'POWER' AND area = ? AND product = ?", (area, load))
GAS = ("commodity = 'NATGAS' AND area = 'TTF' AND product = 'Physical'", ())
EUA = ("short_code = 'FEUA'", ())


def history(where: str, params: tuple) -> pd.DataFrame:
    h = q(f"""SELECT CAST(trade_date AS DATE) AS trade_date, maturity, maturity_type, settlement
              FROM eex_settlements WHERE {where} AND settlement IS NOT NULL ORDER BY trade_date""", params)
    h["contract"] = [contract_label(m, t) for m, t in zip(h["maturity"], h["maturity_type"])]
    return h


def pick(h: pd.DataFrame, day: dt.date, n_month=3, n_quarter=4, n_year=3, decembers=False) -> list:
    """The hedging contracts quoted on `day`: the next months, quarters and years that have not
    started delivering (for carbon: the next December contracts)."""
    today = h[h["trade_date"] == day]
    next_month = (day.year * 100 + day.month) + (89 if day.month == 12 else 1)
    if decembers:
        dec = today[(today["maturity"] % 100 == 12) & (today["maturity"] >= day.year * 100 + day.month)]
        return dec.sort_values("maturity")["contract"].head(n_year).tolist()
    out = []
    for mtype, n in (("Month", n_month), ("Quarter", n_quarter), ("Year", n_year)):
        rows = today[(today["maturity_type"] == mtype) & (today["maturity"] >= next_month)]
        out += rows.sort_values("maturity")["contract"].head(n).tolist()
    return out


def as_of(h: pd.DataFrame, contract: str, day) -> float | None:
    """Settlement of a contract on the last trade date on or before `day`."""
    s = h[(h["contract"] == contract) & (h["trade_date"] <= day)]
    return None if s.empty else s["settlement"].iloc[-1]


def summary(h: pd.DataFrame, contracts: list, day: dt.date) -> pd.DataFrame:
    dates = sorted(h["trade_date"].unique())
    prev = max((d for d in dates if d < day), default=None)
    rows = []
    for c in contracts:
        s = h[h["contract"] == c]
        now = as_of(h, c, day)
        first = s.iloc[0]
        ref = {"1 day": prev, "1 week": day - dt.timedelta(days=7), "4 weeks": day - dt.timedelta(days=28)}
        r = {"Contract": c, "Settlement": now}
        for k, d in ref.items():
            then = None if d is None or d < first["trade_date"] else as_of(h, c, d)
            r[k] = None if then is None else now - then
        base = None if r["1 week"] is None else now - r["1 week"]
        r["1 week %"] = r["1 week"] / base * 100 if base else None
        r["Since"] = now - first["settlement"]
        r["Low"], r["High"] = s["settlement"].min(), s["settlement"].max()
        r["Range position"] = (now - r["Low"]) / (r["High"] - r["Low"]) * 100 if r["High"] > r["Low"] else 50.0
        r["Trend"] = s["settlement"].round(2).tolist()
        r["first"] = first["trade_date"]
        rows.append(r)
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, unit: str):
    num = st.column_config.NumberColumn
    st.dataframe(
        df.drop(columns=["first"]), hide_index=True, width="stretch",
        column_config={
            "Settlement": num(f"Settlement ({unit})", format="%.2f"),
            "1 day": num(format="%+.2f"), "1 week": num(format="%+.2f"), "4 weeks": num(format="%+.2f"),
            "1 week %": num(format="%+.1f%%"),
            "Since": num(f"Since {df['first'].min():%d %b}", format="%+.2f",
                         help="Change since the first settlement collected for the contract."),
            "Low": num(format="%.2f"), "High": num(format="%.2f"),
            "Range position": st.column_config.ProgressColumn(
                "Where in range", format="%.0f%%", min_value=0, max_value=100,
                help="0% = at the lowest settlement since collection started, 100% = at the highest."),
            "Trend": st.column_config.LineChartColumn("Trend since collection started", width="medium"),
        })


def sentence(r: pd.Series, unit: str) -> str:
    if r["1 week"] is None or pd.isna(r["1 week"]):
        return f"**{r['Contract']}** {r['Settlement']:.2f} {unit}"
    word = "up" if r["1 week"] > 0 else "down" if r["1 week"] < 0 else "unchanged"
    move = "" if word == "unchanged" else f" {abs(r['1 week']):.2f} ({r['1 week %']:+.1f}%)"
    return (f"**{r['Contract']}** {r['Settlement']:.2f} {unit}, {word}{move} on the week, "
            f"{r['Range position']:.0f}% of the way from its low to its high since {r['first']:%d %b}")


# ---------------------------------------------------------------- power
ph = history(*POWER)
if ph.empty:
    st.info("No settlements for this area and load yet.")
    st.stop()
day = ph["trade_date"].max()
pw = summary(ph, pick(ph, day), day)

gh, eh = history(*GAS), history(*EUA)
gas_day = gh["trade_date"].max() if not gh.empty else None
eua_day = eh["trade_date"].max() if not eh.empty else None
gs = summary(gh, pick(gh, gas_day, n_month=1, n_quarter=2, n_year=2), gas_day) if gas_day else pd.DataFrame()
es = summary(eh, pick(eh, eua_day, decembers=True, n_year=2), eua_day) if eua_day else pd.DataFrame()

st.subheader(f"This week, as of the {day:%a %d %b %Y} settlement")
lines = []
cal = pw[pw["Contract"].str.startswith("Cal")]
qtr = pw[pw["Contract"].str.startswith("Q")]
for frame in (cal.head(1), qtr.head(1)):
    if not frame.empty:
        lines.append(f"- {area} {load.lower()} " + sentence(frame.iloc[0], "€/MWh"))
if not gs.empty:
    g = gs[gs["Contract"].str.startswith("Cal")]
    if not g.empty:
        lines.append("- TTF gas " + sentence(g.iloc[0], "€/MWh"))
if not es.empty:
    lines.append("- EUA carbon " + sentence(es.iloc[0], "€/t"))
movers = pw.dropna(subset=["1 week"])
if not movers.empty:
    big = movers.loc[movers["1 week"].abs().idxmax()]
    lines.append(f"- Biggest weekly move on the {area} {load.lower()} curve: **{big['Contract']}** "
                 f"{big['1 week']:+.2f} €/MWh ({big['1 week %']:+.1f}%)")
st.markdown("\n".join(lines))

st.subheader(f"{area} {load.lower()} power")
table(pw, "€/MWh")

# ---------------------------------------------------------------- chart: quarters and years
show_contracts = pw[pw["Contract"].str.match(r"^(Q\d|Cal)-")]["Contract"].tolist()
fig = go.Figure()
for i, c in enumerate(show_contracts[:8]):
    s = ph[ph["contract"] == c]
    fig.add_trace(go.Scatter(x=s["trade_date"], y=s["settlement"], name=c, mode="lines",
                             line=dict(color=SERIES[i], width=2.5 if c.startswith("Cal") else 1.8,
                                       dash="solid" if c.startswith("Cal") else "dot"),
                             hovertemplate=f"{c}: %{{y:.2f}} €/MWh<extra></extra>"))
show(fig_defaults(fig, f"{area} {load.lower()}: quarters and calendar years by trade date", "€/MWh", 400),
     ph[ph["contract"].isin(show_contracts)].pivot_table(index="trade_date", columns="contract", values="settlement")
     .reset_index().round(2), key="hedge_hist")

# ---------------------------------------------------------------- drivers
st.subheader("Drivers: gas and carbon")
st.caption("Gas-fired plants set the German power price most of the time, so power moves with gas and carbon. "
           "See the Gas and carbon page for the history.")
if not gs.empty:
    st.markdown(f"**TTF gas** (EEX, as of {gas_day:%d %b %Y})")
    table(gs, "€/MWh")
if not es.empty:
    st.markdown(f"**EUA carbon** (EEX December futures, as of {eua_day:%d %b %Y})")
    table(es, "€/t")
st.caption("A 1-week change compares with the last settlement on or before the same weekday a week earlier. "
           "Changes are blank where the contract has no settlement that far back.")
