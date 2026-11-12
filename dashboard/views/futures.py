import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import INK2, MUTED, SERIES, color_map, fig_defaults, page, q, rebuild_button, show

page("EEX futures", "Daily settlement prices collected from EEX. History starts when the scraper started (Sept 2026).")
rebuild_button()

MONTHS_IN = {"Month": 1, "Quarter": 3, "Year": 12}


def contract_label(maturity, mtype) -> str:
    m = str(maturity)
    start = pd.Timestamp(f"{m[:4]}-{m[4:6]}-01")
    if mtype == "Quarter":
        return f"Q{(start.month - 1) // 3 + 1}-{start:%y}"
    if mtype == "Year":
        return f"Cal-{start:%y}"
    return start.strftime("%b-%y")


# ---------------------------------------------------------------- filters
f = st.columns([1.2, 1, 1, 1.4, 1.4])
market = f[0].segmented_control("Market", ["Power", "Carbon"], default="Power")
if market == "Power":
    areas = q("SELECT DISTINCT area FROM eex_settlements WHERE commodity = 'POWER' ORDER BY area")["area"].tolist()
    area = f[1].selectbox("Area", areas, index=areas.index("DE") if "DE" in areas else 0)
    product = f[2].selectbox("Load", ["Base", "Peak"])
    where = "commodity = 'POWER' AND area = ? AND product = ?"
    params = (area, product)
    unit = "€/MWh"
else:
    product = f[1].selectbox("Product", ["EUA", "UKA", "ETS2"])
    where = "commodity = 'ENVIRONMENTALS' AND product = ?"
    params = (product,)
    unit = "€/t" if product != "UKA" else "£/t"
    area = None

dates = q(f"SELECT DISTINCT trade_date FROM eex_settlements WHERE {where} AND settlement IS NOT NULL ORDER BY trade_date DESC", params)
if dates.empty:
    st.info("No settlements for this selection yet.")
    st.stop()
trade_date = f[3].selectbox("Trade date", dates["trade_date"].tolist(), format_func=lambda d: f"{d:%a %d %b %Y}")
compare_options = ["None"] + [d for d in dates["trade_date"].tolist() if d < trade_date]
compare = f[4].selectbox("Compare with", compare_options,
                         index=min(5, len(compare_options) - 1),
                         format_func=lambda d: d if d == "None" else f"{d:%a %d %b %Y}")

curve = q(f"""SELECT trade_date, maturity, maturity_type, settlement, volume, gross_open_interest_lots AS open_interest
              FROM eex_settlements WHERE {where} AND settlement IS NOT NULL
              AND trade_date IN (?, ?) ORDER BY maturity""",
          params + (trade_date, trade_date if compare == "None" else compare))
curve["contract"] = [contract_label(m, t) for m, t in zip(curve["maturity"], curve["maturity_type"])]
curve["start"] = pd.to_datetime(curve["maturity"].astype(str), format="%Y%m")
curve["end"] = [s + pd.DateOffset(months=MONTHS_IN.get(t, 1)) for s, t in zip(curve["start"], curve["maturity_type"])]

# ---------------------------------------------------------------- forward curve
fig = go.Figure()
types = [t for t in ("Month", "Quarter", "Year") if t in set(curve["maturity_type"])]
cmap = color_map(types)
for t in types:
    for when, dash, width, suffix in ((trade_date, "solid", 3, ""), (compare, "dot", 2, " (compare)")):
        if when == "None" or (suffix and when == trade_date):
            continue
        d = curve[(curve["maturity_type"] == t) & (curve["trade_date"] == when)]
        if d.empty:
            continue
        if market == "Power":
            xs, ys, texts = [], [], []
            for _, r in d.iterrows():
                xs += [r["start"], r["end"], None]
                ys += [r["settlement"], r["settlement"], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cmap[t], width=width, dash=dash),
                                     name=f"{t}{suffix}", hoverinfo="skip", legendgroup=f"{t}{suffix}"))
            mid = d["start"] + (d["end"] - d["start"]) / 2
            fig.add_trace(go.Scatter(x=mid, y=d["settlement"], mode="markers", showlegend=False,
                                     legendgroup=f"{t}{suffix}",
                                     marker=dict(color=cmap[t], size=8 if not suffix else 6, line=dict(color="white", width=1)),
                                     customdata=d[["contract"]], name=f"{t}{suffix}",
                                     hovertemplate=f"%{{customdata[0]}}{suffix}: %{{y:.2f}} {unit}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=d["start"], y=d["settlement"], mode="lines+markers", name=f"{when:%d %b}",
                                     line=dict(color=SERIES[0] if not suffix else MUTED, width=width, dash=dash),
                                     marker=dict(size=8), customdata=d[["contract"]],
                                     hovertemplate=f"%{{customdata[0]}}: %{{y:.2f}} {unit}<extra>{when:%d %b}</extra>"))
title = (f"{area} {product} forward curve" if market == "Power" else f"{product} futures by expiry") + f", {trade_date:%d %b %Y}"
fig.update_layout(hovermode="closest")
if market == "Power":  # open on the liquid part of the curve; drag or double-click to see all of it
    fig.update_xaxes(range=[curve["start"].min() - pd.Timedelta(days=15), curve["start"].min() + pd.DateOffset(years=4)])
show(fig_defaults(fig, title, unit, 440),
     curve[curve["trade_date"] == trade_date][["contract", "maturity_type", "settlement", "volume", "open_interest"]])
if market == "Power":
    st.caption("Each line spans its delivery period; dotted lines show the comparison date. "
               "The chart opens on the next four years: double-click it to see the whole curve.")

# ---------------------------------------------------------------- history of selected contracts
st.subheader("Settlement history")
latest = curve[curve["trade_date"] == trade_date]
choices = latest["contract"].tolist()
default = choices[:3] if market == "Power" else [c for c in choices if c.startswith("Dec")][:3] or choices[:3]
picked = st.multiselect("Contracts (up to 8)", choices, default=default, max_selections=8)
if picked:
    keys = latest[latest["contract"].isin(picked)][["maturity", "maturity_type", "contract"]]
    hist = q(f"""SELECT trade_date, maturity, maturity_type, settlement FROM eex_settlements
                 WHERE {where} AND settlement IS NOT NULL ORDER BY trade_date""", params)
    hist = hist.merge(keys, on=["maturity", "maturity_type"])
    cmap = color_map(picked)
    fig = go.Figure()
    for cname in picked:
        d = hist[hist["contract"] == cname]
        fig.add_trace(go.Scatter(x=d["trade_date"], y=d["settlement"], name=cname, mode="lines",
                                 line=dict(color=cmap[cname], width=2),
                                 hovertemplate=f"{cname}: %{{y:.2f}} {unit}<extra></extra>"))
    show(fig_defaults(fig, "Settlement by trade date", unit, 380),
         hist.pivot_table(index="trade_date", columns="contract", values="settlement").reset_index())
