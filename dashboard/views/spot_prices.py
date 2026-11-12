import plotly.graph_objects as go
import streamlit as st

from lib import color_map, date_range, fig_defaults, page, q, rebuild_button, show

page("Day-ahead prices", "Day-ahead auction prices by bidding zone (Energy-Charts). Days are delivery days in Central European time.")
rebuild_button()

zones = q("SELECT DISTINCT zone FROM prices_daily ORDER BY zone")["zone"].tolist()
bounds = q("SELECT min(day) lo, max(day) hi FROM prices_daily").iloc[0]

f = st.columns([2.6, 1, 1.5, 2.4])
picked = f[0].multiselect("Bidding zones (up to 8)", zones, default=["DE-LU", "FR", "NL", "PL"], max_selections=8)
measure = f[1].segmented_control("Load", ["Base", "Peak"], default="Base")
grain = f[2].segmented_control("Average", ["Day", "Week", "Month"], default="Day")
with f[3]:
    start, end = date_range("spot", 183, bounds["lo"], bounds["hi"])

if not picked:
    st.info("Pick at least one bidding zone.")
    st.stop()

col = (measure or "Base").lower()
trunc = {"Day": "day", "Week": "week", "Month": "month"}[grain or "Day"]
placeholders = ",".join("?" * len(picked))
df = q(f"""SELECT zone, date_trunc('{trunc}', day) AS period, avg({col}) AS price
           FROM prices_daily WHERE zone IN ({placeholders}) AND day BETWEEN ? AND ?
           GROUP BY ALL ORDER BY period""", tuple(picked) + (start, end))
cmap = color_map(picked)  # 16 zones but only 8 hues: colours go by selection order
fig = go.Figure()
for z in picked:
    d = df[df["zone"] == z]
    fig.add_trace(go.Scatter(x=d["period"], y=d["price"], name=z, mode="lines", line=dict(color=cmap[z], width=2),
                             hovertemplate=f"{z}: %{{y:.2f}} €/MWh<extra></extra>"))
show(fig_defaults(fig, f"{measure} price, {dict(Day='daily', Week='weekly', Month='monthly')[grain or 'Day']} average", "€/MWh"),
     df.pivot_table(index="period", columns="zone", values="price").reset_index().round(2), key="spot_main")

left, right = st.columns(2)
with left:
    shape = q(f"""SELECT zone, hour(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ))) AS hour,
                         avg(price_eur_mwh) AS price
                  FROM da_prices WHERE zone IN ({placeholders})
                  AND CAST(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ)) AS DATE) BETWEEN ? AND ?
                  GROUP BY ALL ORDER BY hour""", tuple(picked) + (start, end))
    fig = go.Figure()
    for z in picked:
        d = shape[shape["zone"] == z]
        fig.add_trace(go.Scatter(x=d["hour"], y=d["price"], name=z, mode="lines+markers",
                                 line=dict(color=cmap[z], width=2), marker=dict(size=6),
                                 hovertemplate=f"{z}: %{{y:.2f}} €/MWh<extra></extra>"))
    fig.update_xaxes(dtick=3, title="Hour of day (CET)")
    show(fig_defaults(fig, "Average daily shape over the range", "€/MWh", 360),
         shape.pivot_table(index="hour", columns="zone", values="price").reset_index().round(2), key="spot_shape")
with right:
    neg = q(f"""SELECT zone, sum(negative_intervals) AS neg, sum(intervals) AS n
                FROM prices_daily WHERE zone IN ({placeholders}) AND day BETWEEN ? AND ? GROUP BY zone""",
            tuple(picked) + (start, end))
    neg["share"] = 100 * neg["neg"] / neg["n"]
    neg = neg.set_index("zone").loc[picked].reset_index()
    fig = go.Figure(go.Bar(x=neg["zone"], y=neg["share"], marker_color=[cmap[z] for z in neg["zone"]],
                           marker_cornerradius=4, hovertemplate="%{x}: %{y:.1f}% of intervals<extra></extra>"))
    fig.update_layout(hovermode="closest", bargap=0.45)
    show(fig_defaults(fig, "Share of intervals with negative prices", "%", 360),
         neg.rename(columns={"neg": "negative intervals", "n": "intervals", "share": "share %"}).round(1), key="spot_neg")
