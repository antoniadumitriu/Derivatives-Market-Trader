import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import INK, INK2, MUTED, SERIES, color_map, date_range, fig_defaults, page, q, rebuild_button, show

from pipeline.build_db import CCGT_EFFICIENCY, GAS_EMISSIONS_T_PER_MWH

page("Gas and carbon", "TTF gas and EUA carbon prices, and what they imply for the cost of gas-fired power.")
rebuild_button()

bounds = q("SELECT min(day) lo, max(day) hi FROM fuel_daily WHERE ttf_eur_mwh IS NOT NULL").iloc[0]
f = st.columns([1.2, 2.4, 2])
grain = f[0].segmented_control("Average", ["Day", "Week", "Month"], default="Week")
with f[1]:
    start, end = date_range("fuel", 1096, bounds["lo"], bounds["hi"])
trunc = {"Day": "day", "Week": "week", "Month": "month"}[grain or "Week"]

latest = q("SELECT * FROM fuel_daily WHERE ttf_eur_mwh IS NOT NULL ORDER BY day DESC LIMIT 1").iloc[0]
c = st.columns(3)
c[0].metric("TTF gas front month (€/MWh)", f"{latest['ttf_eur_mwh']:.2f}", help=f"As of {latest['day']:%d %b %Y}. Yahoo Finance.")
c[1].metric("EUA carbon (€/t)", f"{latest['eua_eur_t']:.2f}",
            help="EEX December future. Before Sept 2026: a carbon ETC scaled to the EEX level.")
c[2].metric("Gas plant cost (€/MWh)", f"{latest['gas_srmc_eur_mwh']:.2f}",
            help=f"Fuel and carbon cost of a gas-fired plant at {CCGT_EFFICIENCY:.0%} efficiency: "
                 f"(gas + {GAS_EMISSIONS_T_PER_MWH} t/MWh x carbon) / {CCGT_EFFICIENCY}.")

# ---------------------------------------------------------------- gas plant cost vs power price
d = q(f"""SELECT date_trunc('{trunc}', f.day) AS period, avg(f.ttf_eur_mwh) AS ttf, avg(f.eua_eur_t) AS eua,
                 avg(f.gas_srmc_eur_mwh) AS cost, avg(p.base) AS power,
                 bool_or(f.eua_source = 'proxy') AS proxy
          FROM fuel_daily f LEFT JOIN prices_daily p ON p.day = f.day + INTERVAL 1 DAY AND p.zone = 'DE-LU'
          WHERE f.day BETWEEN ? AND ? GROUP BY ALL ORDER BY period""", (start, end))
fig = go.Figure()
fig.add_trace(go.Scatter(x=d["period"], y=d["power"], name="German day-ahead base", line=dict(color=SERIES[0], width=2),
                         hovertemplate="Power: %{y:.1f} €/MWh<extra></extra>"))
fig.add_trace(go.Scatter(x=d["period"], y=d["cost"], name="Gas plant cost (fuel and carbon)", line=dict(color=SERIES[1], width=2),
                         hovertemplate="Gas plant cost: %{y:.1f} €/MWh<extra></extra>"))
show(fig_defaults(fig, "German power price vs. the cost of gas-fired power", "€/MWh", 420),
     d.round(2), key="spread")
st.caption("When the power price sits near the gas plant cost, gas is setting the price. Well below it, "
           "renewables, coal or imports are. The model uses this cost to anchor its price level.")

left, right = st.columns(2)
with left:
    fig = go.Figure(go.Scatter(x=d["period"], y=d["ttf"], name="TTF", line=dict(color=SERIES[0], width=2),
                               hovertemplate="TTF: %{y:.2f} €/MWh<extra></extra>"))
    show(fig_defaults(fig, "TTF gas, front month", "€/MWh", 360), None, key="ttf")
with right:
    fig = go.Figure()
    proxy, eex = d[d["proxy"]], d[~d["proxy"]]
    fig.add_trace(go.Scatter(x=proxy["period"], y=proxy["eua"], name="Carbon ETC, scaled (proxy)",
                             line=dict(color=SERIES[1], width=2, dash="dot"),
                             hovertemplate="EUA (proxy): %{y:.2f} €/t<extra></extra>"))
    fig.add_trace(go.Scatter(x=eex["period"], y=eex["eua"], name="EEX December future",
                             line=dict(color=SERIES[1], width=2), hovertemplate="EUA: %{y:.2f} €/t<extra></extra>"))
    show(fig_defaults(fig, "EUA carbon", "€/t", 360), None, key="eua")

# ---------------------------------------------------------------- TTF forward curve
st.subheader("TTF forward curve")
curve = q("""SELECT CAST(trade_date AS DATE) AS trade_date, CAST(delivery_month AS VARCHAR) AS month, close
             FROM ttf_curve ORDER BY trade_date""")
eex_gas = q("""SELECT trade_date, area, maturity, maturity_type, settlement FROM eex_settlements
               WHERE commodity = 'NATGAS' AND maturity_type = 'Month' AND settlement IS NOT NULL""")
if not eex_gas.empty:
    st.caption("EEX TTF and THE gas settlements are now collected daily; see the EEX futures page for the full curves.")
# Only dates with at least 3 contracts priced (Yahoo fills some contracts later in the day).
counts = curve.groupby("trade_date")["month"].nunique()
dates = sorted(counts[counts >= min(3, counts.max())].index, reverse=True)
if dates:
    cc = st.columns([1.5, 1.5, 3])
    d1 = cc[0].selectbox("Trade date", dates, format_func=lambda x: f"{pd.Timestamp(x):%a %d %b %Y}")
    older = [x for x in dates if x < d1]
    d0 = cc[1].selectbox("Compare with", ["None"] + older, index=min(5, len(older)),
                         format_func=lambda x: x if x == "None" else f"{pd.Timestamp(x):%a %d %b %Y}")
    fig = go.Figure()
    for when, dash, color in ((d1, "solid", SERIES[0]), (d0, "dot", MUTED)):
        if when == "None":
            continue
        k = curve[curve["trade_date"] == when].sort_values("month")
        k["label"] = pd.to_datetime(k["month"], format="%Y%m").dt.strftime("%b %y")
        fig.add_trace(go.Scatter(x=k["label"], y=k["close"], mode="lines+markers", name=f"{pd.Timestamp(when):%d %b %Y}",
                                 line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
                                 hovertemplate="%{x}: %{y:.2f} €/MWh<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    show(fig_defaults(fig, "TTF by delivery month (Yahoo Finance)", "€/MWh", 360),
         curve[curve["trade_date"] == d1], key="ttfcurve")
    st.caption("Yahoo lists only the nearer TTF contracts. The model uses these prices for the months it forecasts.")
