import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import DATA, INK2, MODEL, OBSERVED, fig_defaults, page, q, rebuild_button, show, ttf_quote

page("Overview", "European power and carbon: headline numbers and data health. Prices in €/MWh unless noted.")
rebuild_button()

# ---------------------------------------------------------------- headline numbers
latest = q("""
    SELECT day, base, peak FROM prices_daily WHERE zone = 'DE-LU' ORDER BY day DESC LIMIT 2""")
front = q("""
    SELECT trade_date, month, settlement FROM de_base_month_futures
    WHERE months_ahead = 1 ORDER BY trade_date DESC LIMIT 2""")
eua = q("""
    SELECT trade_date, maturity, settlement FROM eex_settlements
    WHERE short_code = 'FEUA' AND settlement IS NOT NULL
      AND maturity = (SELECT min(maturity) FROM eex_settlements
                      WHERE short_code = 'FEUA' AND settlement IS NOT NULL
                        AND CAST(maturity AS VARCHAR) LIKE '%12'
                        AND trade_date = (SELECT max(trade_date) FROM eex_settlements WHERE short_code = 'FEUA'))
    ORDER BY trade_date DESC LIMIT 2""")
month_now = q("SELECT month, base_index, days FROM de_month_index ORDER BY month DESC LIMIT 1")
model = None
try:
    import json

    from lib import REPORTS, model_file
    model = pd.read_csv(model_file(REPORTS, "forecast.csv"), dtype={"month": str}).iloc[0]
    model_through = json.loads(model_file(REPORTS, "params.json").read_text())["origin"]
    # Compare with the latest settlement for that month, not the one stored with the forecast.
    latest_fut = q("""SELECT trade_date, settlement FROM de_base_month_futures WHERE month = ?
                      ORDER BY trade_date DESC LIMIT 1""", (model["month"],))
except FileNotFoundError:
    pass


def delta(df, col):
    return None if len(df) < 2 else f"{df[col].iloc[0] - df[col].iloc[1]:+.2f}"


def mlabel(m) -> str:
    m = str(m)
    return pd.Timestamp(f"{m[:4]}-{m[4:6]}-01").strftime("%b %y")


c = st.columns(6)
c[0].metric(f"DE spot base {latest['day'].iloc[0]:%d %b}", f"{latest['base'].iloc[0]:.2f}",
            delta(latest, "base"), delta_color="off", help="Average of the day-ahead auction for the delivery day. "
            "Change is against the previous day.")
c[1].metric(f"DE spot {mlabel(month_now['month'].iloc[0])} to date",
            f"{month_now['base_index'].iloc[0]:.2f}", help=f"Average over {month_now['days'].iloc[0]} delivery days so far.")
c[2].metric(f"EEX DE base {mlabel(front['month'].iloc[0])}",
            f"{front['settlement'].iloc[0]:.2f}", delta(front, "settlement"), delta_color="off",
            help=f"Settlement on {front['trade_date'].iloc[0]:%d %b %Y}. Change is against the previous trading day.")
c[3].metric(f"EEX EUA Dec-{str(eua['maturity'].iloc[0])[2:4]} (€/t)", f"{eua['settlement'].iloc[0]:.2f}",
            delta(eua, "settlement"), delta_color="off",
            help=f"Settlement on {eua['trade_date'].iloc[0]:%d %b %Y}.")
ttf = ttf_quote()
c[4].metric("TTF gas front month" + (" (intraday)" if ttf["intraday"] else ""), f"{ttf['value']:.2f}",
            None if ttf["prev"] is None else f"{ttf['value'] - ttf['prev']:+.2f}", delta_color="off",
            help=f"{ttf['label']}. Change is against the previous trading day. See the Gas and carbon page.")
if model is not None:
    fut_now = latest_fut["settlement"].iloc[0] if len(latest_fut) else model["futures"]
    fut_day = latest_fut["trade_date"].iloc[0] if len(latest_fut) else None
    c[5].metric(f"Model DE base {mlabel(model['month'])}", f"{model['mean']:.2f}",
                f"{model['mean'] - fut_now:+.2f} vs futures", delta_color="off",
                help=f"Posterior mean of the current Bayesian month model, fitted on data through {model_through}. "
                     + (f"Compared with the EEX settlement of {fut_day:%d %b %Y} ({fut_now:.2f}). " if fut_day else "")
                     + "See the model page.")
    age = (pd.Timestamp(latest["day"].iloc[0]) - pd.Timestamp(model_through)).days
    if age > 3:
        st.warning(f"The model forecast is {age} days old (data through {model_through}). The daily refit may "
                   "have failed: check the Model daily workflow on GitHub, or refit on the model page.")

# ---------------------------------------------------------------- price chart
hist = q("""SELECT day, base FROM prices_daily WHERE zone = 'DE-LU' AND day >= current_date - INTERVAL 120 DAY ORDER BY day""")
months = q("""SELECT month, base_index FROM de_month_index ORDER BY month DESC LIMIT 5""")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["day"], y=hist["base"], name="Daily base", line=dict(color=MODEL, width=2),
                         hovertemplate="%{y:.2f} €/MWh"))
for _, r in months.iterrows():
    start = max(pd.Timestamp(f"{r['month'][:4]}-{r['month'][4:]}-01"), pd.Timestamp(hist["day"].min()))
    end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp(hist["day"].max()))
    if end < pd.Timestamp(hist["day"].min()):
        continue
    fig.add_trace(go.Scatter(x=[start, end], y=[r["base_index"]] * 2, mode="lines", name="Monthly average",
                             line=dict(color=OBSERVED, width=2), showlegend=bool(_ == 0),
                             hovertemplate=f"{mlabel(r['month'])} average: %{{y:.2f}} €/MWh<extra></extra>"))
show(fig_defaults(fig, "German day-ahead base price, last 120 days", "€/MWh", 360), hist)

# ---------------------------------------------------------------- data health
st.subheader("Data health")
now = pd.Timestamp.now(tz="UTC").tz_localize(None)
health = q("""
    SELECT 'EEX futures settlements' AS dataset, CAST(max(trade_date) AS TIMESTAMP) AS latest, count(*) AS rows FROM eex_settlements WHERE settlement IS NOT NULL
    UNION ALL SELECT 'Day-ahead prices', max(CAST(timestamp_utc AS TIMESTAMP)), count(*) FROM da_prices
    UNION ALL SELECT 'Generation', max(CAST(timestamp_utc AS TIMESTAMP)), count(*) FROM generation
    UNION ALL SELECT 'Power forecasts', max(CAST(timestamp_utc AS TIMESTAMP)), count(*) FROM power_forecasts
    UNION ALL SELECT 'Cross-border flows', max(CAST(timestamp_utc AS TIMESTAMP)), count(*) FROM crossborder
    UNION ALL SELECT 'Weather actuals', max(CAST(timestamp_utc AS TIMESTAMP)), count(*) FROM weather_actuals
    UNION ALL SELECT 'Weather forecasts (issued)', max(CAST(issue_date AS TIMESTAMP)), count(*) FROM weather_forecasts
    UNION ALL SELECT 'Gas, carbon proxy, EUR/USD (Yahoo)', max(CAST(date AS TIMESTAMP)), count(*) FROM market_prices
    UNION ALL SELECT 'TTF forward curve (Yahoo)', max(CAST(trade_date AS TIMESTAMP)), count(*) FROM ttf_curve
""")
log_path = DATA / "model" / "forecast_log.csv"
if log_path.exists():
    flog = pd.read_csv(log_path)
    health = pd.concat([health, pd.DataFrame([{"dataset": "Model forecasts (last refit)",
                                                "latest": pd.Timestamp(flog["run_utc"].max()).tz_localize(None),
                                                "rows": len(flog)}])], ignore_index=True)
runs = []
for name, table, okcol, failcol in (("EEX scraper", "runs_eex", "ok", "failed"),
                                    ("Energy-Charts", "runs_energy_charts", "ok", "failed"),
                                    ("Yahoo Finance (gas, carbon)", "runs_market", "ok", "failed"),
                                    ("Open-Meteo", "runs_weather", None, None)):
    r = q(f"SELECT * FROM {table} ORDER BY run_utc DESC LIMIT 1").iloc[0]
    if okcol:
        ok, failed = int(r[okcol]), int(r[failcol])
        state = "good" if failed == 0 else "warning" if failed <= 0.2 * (ok + failed) else "critical"
        detail = f"{ok} requests ok, {failed} failed"
    else:
        err = r.get("errors")
        state = "good" if pd.isna(err) or not str(err).strip() else "critical"
        detail = "no errors" if state == "good" else str(err)[:120]
    runs.append({"source": name, "last run (UTC)": str(r["run_utc"])[:16].replace("T", " "), "status": state, "detail": detail})

left, right = st.columns([3, 2])
with left:
    st.caption("Latest data point per dataset")
    health["latest"] = pd.to_datetime(health["latest"])
    health["age"] = (now - health["latest"]).apply(
        lambda d: "in the future (forecast or next-day auction)" if d.total_seconds() < 0
        else f"{d.days} d {d.seconds // 3600} h ago")
    health["latest"] = health["latest"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(health, hide_index=True, width="stretch",
                 column_config={"rows": st.column_config.NumberColumn(format="localized")})
with right:
    st.caption("Last collection run per source")
    icon = {"good": "✅ OK", "warning": "⚠️ Some failures", "critical": "⛔ Failed"}
    for r in runs:
        st.markdown(f"**{r['source']}**: {icon[r['status']]}  \n"
                    f"<span style='color:{INK2}'>{r['last run (UTC)']} UTC, {r['detail']}</span>",
                    unsafe_allow_html=True)
    if log_path.exists():
        last = flog.sort_values("run_utc").iloc[-1]
        stale = (now - pd.Timestamp(last["run_utc"]).tz_localize(None)).days > 3
        st.markdown(f"**Model refit**: {'⚠️ Over 3 days old' if stale else '✅ OK'}  \n"
                    f"<span style='color:{INK2}'>{str(last['run_utc'])[:16].replace('T', ' ')} UTC, "
                    f"data through {last['data_through']}</span>", unsafe_allow_html=True)
    st.caption("Runs happen on GitHub Actions. Pull the data repo to see new data here.")
