import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import INK, SERIES, color_map, date_range, fig_defaults, page, q, rebuild_button, show

page("Fundamentals", "Generation, load, forecasts, cross-border flows and capacity (Energy-Charts). Daily averages, GW. A day with any hour missing is left blank rather than averaged over the hours that are there.")
rebuild_button()

COUNTRIES = {"de": "Germany", "fr": "France", "nl": "Netherlands", "be": "Belgium",
             "at": "Austria", "ch": "Switzerland", "pl": "Poland", "cz": "Czechia"}
ZONE = {"de": "DE-LU", "fr": "FR", "nl": "NL", "be": "BE", "at": "AT", "ch": "CH", "pl": "PL", "cz": "CZ"}
# Seven technology groups, in a fixed order (stack bottom to top).
GROUPS = {
    "Nuclear": ["nuclear"],
    "Coal and lignite": ["fossil_brown_coal_lignite", "fossil_hard_coal", "fossil_coal_derived_gas"],
    "Gas": ["fossil_gas"],
    "Hydro": ["hydro_run_of_river", "hydro_water_reservoir", "hydro_pumped_storage"],
    "Biomass and other": ["biomass", "geothermal", "waste", "others", "fossil_oil", "other_renewables"],
    "Wind": ["wind_onshore", "wind_offshore"],
    "Solar": ["solar"],
}
CAP_LABELS = {"wind_onshore": "Wind onshore", "wind_offshore": "Wind offshore", "solar_ac": "Solar",
              "solar_dc": "Solar", "solar": "Solar", "battery_storage_power": "Batteries (power)",
              "fossil_gas": "Gas", "fossil_brown_coal_lignite": "Lignite", "fossil_hard_coal": "Hard coal",
              "nuclear": "Nuclear"}
GROUP_COLOR = dict(zip(GROUPS, ["#6250d6", "#52514e", "#eb6834", "#1baf7a", "#008300", "#2a78d6", "#eda100"]))

bounds = q("SELECT min(day) lo, max(day) hi FROM generation_daily").iloc[0]
f = st.columns([1.3, 1, 2])
country = f[0].selectbox("Country", list(COUNTRIES), format_func=COUNTRIES.get)
grain = f[1].segmented_control("Average", ["Day", "Week", "Month"], default="Week")
with f[2]:
    start, end = date_range("fund", 365, bounds["lo"], bounds["hi"])
trunc = {"Day": "day", "Week": "week", "Month": "month"}[grain or "Week"]

cols = [c for c in q("SELECT * FROM generation_daily LIMIT 0").columns]
exprs = []
for g, members in GROUPS.items():
    present = [m for m in members if m in cols]
    exprs.append(f"avg({' + '.join(f'coalesce({m}, 0)' for m in present) or '0'}) / 1000 AS \"{g}\"")
gen = q(f"""SELECT date_trunc('{trunc}', day) AS period, {', '.join(exprs)},
                   avg(load) / 1000 AS "Load", avg(residual_load) / 1000 AS "Residual load"
            FROM generation_daily WHERE country = ? AND day BETWEEN ? AND ? GROUP BY ALL ORDER BY period""",
        (country, start, end))

# ---------------------------------------------------------------- generation mix
fig = go.Figure()
for g in GROUPS:
    if gen[g].abs().sum() < 1e-6:
        continue
    fig.add_trace(go.Scatter(x=gen["period"], y=gen[g], name=g, stackgroup="gen", mode="lines",
                             line=dict(width=0.5, color=GROUP_COLOR[g]), fillcolor=GROUP_COLOR[g],
                             hovertemplate=f"{g}: %{{y:.1f}} GW<extra></extra>"))
fig.add_trace(go.Scatter(x=gen["period"], y=gen["Load"], name="Load", mode="lines",
                         line=dict(color=INK, width=2), hovertemplate="Load: %{y:.1f} GW<extra></extra>"))
show(fig_defaults(fig, f"{COUNTRIES[country]}: net generation by technology and load", "GW", 460), gen.round(2), key="mix")
st.caption("Generation can sit above or below load: the gap is net exports or imports, plus storage.")

left, right = st.columns(2)

# ---------------------------------------------------------------- merit order: price vs residual load
with left:
    sc = q("""SELECT g.day, year(g.day) AS year, g.residual_load / 1000 AS rl, p.base
              FROM generation_daily g JOIN prices_daily p ON p.day = g.day AND p.zone = ?
              WHERE g.country = ? AND g.day BETWEEN ? AND ?""", (ZONE[country], country, start, end))
    years = sorted(sc["year"].unique())[-8:]
    cmap = color_map(years)
    fig = go.Figure()
    for y in years:
        d = sc[sc["year"] == y]
        fig.add_trace(go.Scatter(x=d["rl"], y=d["base"], mode="markers", name=str(y),
                                 marker=dict(color=cmap[y], size=8, opacity=0.75, line=dict(color="white", width=1)),
                                 customdata=d[["day"]],
                                 hovertemplate="%{customdata[0]|%d %b %Y}: %{x:.1f} GW, %{y:.2f} €/MWh<extra></extra>"))
    fig.update_layout(hovermode="closest", xaxis_title="Residual load, GW (load minus wind and solar)")
    show(fig_defaults(fig, f"Day-ahead base price vs. residual load ({ZONE[country]})", "€/MWh", 420),
         sc.round(2), key="merit")

# ---------------------------------------------------------------- forecast vs actual
with right:
    fc = q(f"""SELECT date_trunc('{trunc}', f.day) AS period,
                      avg(f.solar_day_ahead) / 1000 AS solar_fc, avg(g.solar) / 1000 AS solar,
                      avg(coalesce(f.wind_onshore_day_ahead, 0) + coalesce(f.wind_offshore_day_ahead, 0)) / 1000 AS wind_fc,
                      avg(coalesce(g.wind_onshore, 0) + coalesce(g.wind_offshore, 0)) / 1000 AS wind
               FROM power_forecasts_daily f JOIN generation_daily g USING (country, day)
               WHERE country = ? AND f.day BETWEEN ? AND ? GROUP BY ALL ORDER BY period""", (country, start, end))
    fig = go.Figure()
    for name, actual, fcst, color in (("Wind", "wind", "wind_fc", GROUP_COLOR["Wind"]),
                                      ("Solar", "solar", "solar_fc", GROUP_COLOR["Solar"])):
        if fc[fcst].fillna(0).abs().sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=fc["period"], y=fc[actual], name=f"{name} actual", mode="lines",
                                 line=dict(color=color, width=2), hovertemplate=f"{name} actual: %{{y:.1f}} GW<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["period"], y=fc[fcst], name=f"{name} day-ahead forecast", mode="lines",
                                 line=dict(color=color, width=2, dash="dot"),
                                 hovertemplate=f"{name} forecast: %{{y:.1f}} GW<extra></extra>"))
    show(fig_defaults(fig, "Wind and solar: day-ahead forecast vs. actual", "GW", 420), fc.round(2), key="fc")

left, right = st.columns(2)

# ---------------------------------------------------------------- cross-border flows
with left:
    cb_cols = [c for c in q("SELECT * FROM crossborder_daily LIMIT 0").columns if c not in ("country", "day", "sum")]
    avg_exprs = ", ".join(f"avg({c}) AS {c}" for c in cb_cols)
    cb = q(f"SELECT {avg_exprs} FROM crossborder_daily WHERE country = ? AND day BETWEEN ? AND ?",
           (country, start, end)).T.reset_index()
    cb.columns = ["neighbour", "gw"]
    cb = cb.dropna()
    cb = cb[cb["gw"].abs() > 0.005].sort_values("gw")
    cb["neighbour"] = cb["neighbour"].str.replace("_", " ").str.title()
    imports, exports = cb[cb["gw"] >= 0], cb[cb["gw"] < 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=imports["neighbour"], x=imports["gw"], orientation="h", name="Net import",
                         marker_color=SERIES[0], marker_cornerradius=4,
                         hovertemplate="%{y}: imports %{x:.2f} GW<extra></extra>"))
    fig.add_trace(go.Bar(y=exports["neighbour"], x=exports["gw"], orientation="h", name="Net export",
                         marker_color=SERIES[1], marker_cornerradius=4,
                         hovertemplate="%{y}: exports %{customdata:.2f} GW<extra></extra>", customdata=-exports["gw"]))
    fig.update_layout(hovermode="closest", barmode="relative", xaxis_title="GW, average over the range")
    show(fig_defaults(fig, f"{COUNTRIES[country]}: average physical flow by neighbour", "", 420), cb.round(3), key="cb")

# ---------------------------------------------------------------- installed capacity
with right:
    cap = q("""SELECT period, technology, capacity_gw FROM installed_capacity
               WHERE country = ? AND time_step = 'yearly' ORDER BY period""", (country,))
    keep = ["wind_onshore", "wind_offshore", "solar_ac", "solar_dc", "solar", "battery_storage_power",
            "fossil_gas", "fossil_brown_coal_lignite", "fossil_hard_coal", "nuclear"]
    techs = [t for t in keep if t in set(cap["technology"])]
    techs = [t for t in techs if not (t == "solar_dc" and "solar_ac" in techs)][:8]
    cmap = color_map(techs)
    fig = go.Figure()
    for t in techs:
        d = cap[cap["technology"] == t]
        label = CAP_LABELS.get(t, t.replace("_", " ").capitalize())
        fig.add_trace(go.Scatter(x=d["period"].astype(str), y=d["capacity_gw"], name=label, mode="lines+markers",
                                 line=dict(color=cmap[t], width=2), marker=dict(size=7),
                                 hovertemplate=f"{label}: %{{y:.1f}} GW<extra></extra>"))
    show(fig_defaults(fig, f"{COUNTRIES[country]}: installed capacity", "GW", 420),
         cap[cap["technology"].isin(techs)].pivot_table(index="period", columns="technology", values="capacity_gw").reset_index(),
         key="cap")
