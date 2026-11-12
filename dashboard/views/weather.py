import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import GRID, INK2, MODEL, color_map, date_range, fig_defaults, page, q, rebuild_button, show

page("Weather", "Open-Meteo reanalysis actuals and the latest 16-day forecast at the 12 tracked points. Daily values need every hour of the day, so today shows as forecast only.")
rebuild_button()

VARS = {"temperature_2m": ("Temperature", "°C"), "wind_speed_100m": ("Wind speed at 100 m", "m/s"),
        "shortwave_radiation": ("Solar radiation", "W/m²"), "cloud_cover": ("Cloud cover", "%")}

points = q("SELECT point, country, purpose FROM weather_points ORDER BY point")
bounds = q("SELECT min(day) lo, max(day) hi FROM weather_daily").iloc[0]
issue = pd.Timestamp(q("SELECT max(issue_date) AS d FROM weather_forecasts")["d"].iloc[0]).date()


def pname(p: str) -> str:
    country, _, place = p.partition("_")
    return f"{place.replace('_', ' ').title()} ({country.upper()})"

f = st.columns([3, 1.4, 2])
picked = f[0].multiselect("Points (up to 8)", points["point"].tolist(),
                          default=["de_berlin", "de_munich", "de_north_sea"], max_selections=8,
                          format_func=pname)
var = f[1].selectbox("Variable", list(VARS), format_func=lambda v: VARS[v][0])
with f[2]:
    start, end = date_range("wx", 92, bounds["lo"], bounds["hi"])
label, unit = VARS[var]

if picked:
    ph = ",".join("?" * len(picked))
    act = q(f"SELECT point, day, {var} AS v FROM weather_daily WHERE point IN ({ph}) AND day BETWEEN ? AND ? ORDER BY day",
            tuple(picked) + (start, end))
    fc = q(f"""SELECT point, CAST(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ)) AS DATE) AS day, avg({var}) AS v
               FROM weather_forecasts WHERE issue_date = ? AND point IN ({ph}) GROUP BY ALL ORDER BY day""",
           (issue,) + tuple(picked))
    cmap = color_map(picked)
    fig = go.Figure()
    for p in picked:
        name = pname(p)
        a, b = act[act["point"] == p], fc[fc["point"] == p]
        fig.add_trace(go.Scatter(x=a["day"], y=a["v"], name=name, mode="lines", line=dict(color=cmap[p], width=2),
                                 legendgroup=p, hovertemplate=f"{name}: %{{y:.1f}} {unit}<extra></extra>"))
        fig.add_trace(go.Scatter(x=b["day"], y=b["v"], name=f"{name} forecast", mode="lines", legendgroup=p,
                                 showlegend=False, line=dict(color=cmap[p], width=2, dash="dot"),
                                 hovertemplate=f"{name} forecast: %{{y:.1f}} {unit}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp(issue), line=dict(color=GRID, width=2))
    show(fig_defaults(fig, f"{label}, daily average (dotted: forecast issued {issue:%d %b %Y})", unit, 440),
         act.pivot_table(index="day", columns="point", values="v").reset_index().round(2), key="wx")
else:
    st.info("Pick at least one point.")

# ---------------------------------------------------------------- German temperature vs normal
st.subheader("German demand-weighted temperature vs. normal")
t = q("SELECT day, temp_c FROM de_daily WHERE temp_c IS NOT NULL ORDER BY day")
t["day"] = pd.to_datetime(t["day"])
t["doy"] = t["day"].dt.dayofyear.clip(upper=365)
clim = t[t["day"].dt.year < t["day"].dt.year.max()].groupby("doy")["temp_c"]
norm = pd.DataFrame({"mean": clim.mean(), "lo": clim.quantile(0.1), "hi": clim.quantile(0.9)})
norm = norm.rolling(15, center=True, min_periods=1).mean()
recent = t[t["day"] >= t["day"].max() - pd.Timedelta(days=180)].join(norm, on="doy")
fig = go.Figure()
fig.add_trace(go.Scatter(x=recent["day"], y=recent["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=recent["day"], y=recent["lo"], fill="tonexty", fillcolor="rgba(138,137,132,0.18)",
                         line=dict(width=0), name="Normal range (10th to 90th percentile)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=recent["day"], y=recent["mean"], name="Normal", line=dict(color=INK2, width=1.5, dash="dot"),
                         hovertemplate="Normal: %{y:.1f} °C<extra></extra>"))
fig.add_trace(go.Scatter(x=recent["day"], y=recent["temp_c"], name="Actual", line=dict(color=MODEL, width=2),
                         hovertemplate="Actual: %{y:.1f} °C<extra></extra>"))
show(fig_defaults(fig, "Average of six German demand centres, last 180 days", "°C", 380),
     recent[["day", "temp_c", "mean", "lo", "hi"]].round(1), key="norm")
st.caption("Normal is the same calendar day in previous complete years (2021 onwards), smoothed over 15 days.")
