import json
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import DATA, INK, INK2, MODEL, OBSERVED, ROOT, model_file, fig_defaults, page, q, rebuild_button, show

page("Model: German base month", "Bayesian model of the monthly German base price index, compared with EEX futures.")
rebuild_button()

R = ROOT / "reports"
# label: (version id, report folder, refit command)
ALL_VERSIONS = {
    "Version 3: today's solar + calendar month": ("v3_both", R / "de_base_month_v3_both",
                                                  ["models.de_base_month_v3", "--variant", "both"]),
    "Version 2: gas and carbon": ("v2", R / "de_base_month_v2", ["models.de_base_month_v2"]),
    "Version 1: price level only": ("v1", R / "de_base_month", ["models.de_base_month"]),
    "Variant: v2 + today's solar": ("v3_solar", R / "de_base_month_v3_solar",
                                    ["models.de_base_month_v3", "--variant", "solar"]),
    "Variant: v2 + calendar month": ("v3_seasonal", R / "de_base_month_v3_seasonal",
                                     ["models.de_base_month_v3", "--variant", "seasonal"]),
}
VERSIONS = {k: v for k, v in ALL_VERSIONS.items() if (v[1] / "forecast.csv").exists()}
choice = st.segmented_control("Model", list(VERSIONS), default=list(VERSIONS)[0])
version, REPORTS, module = VERSIONS[choice or list(VERSIONS)[0]]
if not (REPORTS / "forecast.csv").exists():
    st.warning(f"No model outputs yet. Run `python -m {' '.join(module)} --backtest 24` first.")
    st.stop()

fc = pd.read_csv(model_file(REPORTS, "forecast.csv"), dtype={"month": str})
params = json.loads(model_file(REPORTS, "params.json").read_text())
states = pd.read_csv(model_file(REPORTS, "states.csv"), parse_dates=["week_start"])
live = model_file(REPORTS, "forecast.csv").parent != REPORTS
bt = pd.read_csv(REPORTS / "backtest.csv", dtype={"target": str}) if (REPORTS / "backtest.csv").exists() else None


def mlabel(m) -> str:
    return pd.Timestamp(f"{m[:4]}-{m[4:6]}-01").strftime("%b %y")


with st.sidebar:
    st.markdown(f"**Refit model {version}**")
    st.caption("Refits on the latest data and refreshes the forecast (about 3 minutes). "
               "The 24-month backtest takes about 20 minutes and runs from the command line.")
    if st.button("Refit now"):
        with st.status("Fitting the model ...", expanded=True) as status:
            for cmd in ([sys.executable, "-m", *module],
                        [sys.executable, "-m", "models.report_de_base_month", "--model", version]):
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode:
                    st.code(r.stderr[-2000:])
                    status.update(label="Refit failed", state="error")
                    st.stop()
            status.update(label="Refit done", state="complete")
        st.rerun()

st.caption(f"Fitted on data through {params['origin']} ({params['weeks']} weeks, "
           f"{params['divergences']} sampler divergences). EEX settlements from {fc['futures_trade_date'].iloc[0]}. "
           + ("From the daily refit on GitHub." if live else "From the last manual run."))
latest_day = q("SELECT max(day) AS d FROM de_daily WHERE residual_load_gw IS NOT NULL")["d"].iloc[0]
age = (pd.Timestamp(latest_day) - pd.Timestamp(params["origin"])).days
if age > 3:
    st.warning(f"This forecast is {age} days behind the data (data through {params['origin']}, latest complete "
               f"day {pd.Timestamp(latest_day):%d %b}). Refit it with the button in the sidebar, or check the "
               "Model daily workflow on GitHub.")


def intervals(fig, x, df, observed, observed_name, row=None, col=None):
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_trace(go.Bar(x=x, y=df["p95"] - df["p05"], base=df["p05"], name="Model 90% interval", width=0.18,
                         marker_color=MODEL, opacity=0.3, marker_cornerradius=4, hoverinfo="skip"), **kw)
    fig.add_trace(go.Bar(x=x, y=df["p75"] - df["p25"], base=df["p25"], name="Model 50% interval", width=0.18,
                         marker_color=MODEL, marker_cornerradius=4, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=df["p50"], mode="markers", name="Model median",
                             marker=dict(symbol="line-ew", size=22, line=dict(color=INK, width=2.5)),
                             customdata=df[["p05", "p25", "p75", "p95"]],
                             hovertemplate="Median %{y:.1f}<br>50%: %{customdata[1]:.1f} to %{customdata[2]:.1f}"
                                           "<br>90%: %{customdata[0]:.1f} to %{customdata[3]:.1f}<extra>Model</extra>"), **kw)
    fig.add_trace(go.Scatter(x=x, y=observed, mode="markers", name=observed_name,
                             marker=dict(color=OBSERVED, size=11, line=dict(color="white", width=2)),
                             hovertemplate=f"{observed_name}: %{{y:.1f}}<extra></extra>"), **kw)
    fig.update_layout(barmode="overlay", hovermode="closest")


# ---------------------------------------------------------------- forward months vs futures
st.subheader("Next months: model vs. EEX futures")
cols = st.columns(len(fc))
for c, (_, r) in zip(cols, fc.iterrows()):
    c.metric(f"DE base {mlabel(r['month'])}", f"{r['mean']:.1f} €/MWh", f"{r['mean'] - r['futures']:+.1f} vs futures {r['futures']:.1f}",
             delta_color="off",
             help=f"Model 90% interval {r['p05']:.0f} to {r['p95']:.0f}. "
                  f"Probability the month settles below the futures price: {r['p_futures_above_model']:.0%}.")
fig = go.Figure()
x = [mlabel(m) for m in fc["month"]]
intervals(fig, x, fc, fc["futures"], "EEX settlement")
show(fig_defaults(fig, "Delivery months ahead", "€/MWh", 400),
     fc[["month", "mean", "p05", "p25", "p50", "p75", "p95", "futures", "p_futures_above_model"]].round(2), key="fwd")
if "ttf_forward" in fc:
    st.caption("Gas used for each month: " + ", ".join(
        f"{mlabel(r['month'])} TTF {r['ttf_forward']:.2f} €/MWh ({r['ttf_source']})" for _, r in fc.iterrows())
        + f"; EUA {fc['eua'].iloc[0]:.2f} €/t; TTF volatility {fc['gas_vol'].iloc[0]:.0%} a year.")
st.caption("Probability below futures near 50% means model and market agree. Well above 50% means the model sees "
           "futures as expensive, well below means cheap." + (" Version 1 does not see gas prices." if version == "v1" else ""))

# ---------------------------------------------------------------- forecast log
log_path = DATA / "model" / "forecast_log.csv"
if log_path.exists():
    log = pd.read_csv(log_path, dtype={"month": str})
    log = log[log["model"] == REPORTS.name]
    if not log.empty:
        st.subheader("Forecast log: model vs. futures over time")
        log["run_day"] = pd.to_datetime(log["data_through"])
        log = log.sort_values("run_utc").drop_duplicates(["data_through", "month"], keep="last")
        months = sorted(log["month"].unique())
        pick_month = st.selectbox("Delivery month", months, index=len(months) - 1 if len(months) < 3 else 0,
                                  format_func=mlabel, key="log_month")
        d = log[log["month"] == pick_month].sort_values("run_day")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(d["run_day"]) + list(d["run_day"][::-1]),
                                 y=list(d["p95"]) + list(d["p05"][::-1]), fill="toself", mode="lines",
                                 fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="Model 90% interval",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["run_day"], y=d["mean"], name="Model mean", mode="lines+markers",
                                 line=dict(color=MODEL, width=2), marker=dict(size=7),
                                 hovertemplate="Model: %{y:.1f} €/MWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["run_day"], y=d["futures"], name="EEX settlement", mode="lines+markers",
                                 line=dict(color=OBSERVED, width=2), marker=dict(size=7),
                                 hovertemplate="Futures: %{y:.1f} €/MWh<extra></extra>"))
        actual = q("SELECT base_index, complete FROM de_month_index WHERE month = ?", (pick_month,))
        if len(actual) and bool(actual["complete"].iloc[0]):
            fig.add_hline(y=float(actual["base_index"].iloc[0]), line=dict(color=INK, width=1.5, dash="dash"),
                          annotation_text=f"Realised {float(actual['base_index'].iloc[0]):.1f}",
                          annotation_position="top left")
        fig.update_xaxes(title="Forecast made with data through")
        show(fig_defaults(fig, f"DE base {mlabel(pick_month)}: each day's forecast and the futures price", "€/MWh", 380),
             d[["data_through", "mean", "p05", "p95", "futures", "futures_trade_date", "p_futures_above_model"]].round(2),
             key="fclog")
        st.caption(f"One point per daily refit since {log['run_day'].min():%d %b %Y}. Once months settle, this is "
                   "the out-of-sample record of whether the model or the futures were closer.")

# ---------------------------------------------------------------- backtest
if bt is not None:
    st.subheader(f"Backtest: {len(bt)} one-month-ahead forecasts")
    m = st.columns(4)
    m[0].metric("Mean absolute error", f"{bt['abs_err_mean'].mean():.1f} €/MWh",
                f"naive {bt['abs_err_naive'].mean():.1f}", delta_color="off",
                help="Naive forecast: average price of the 30 days before the forecast date.")
    m[1].metric("50% interval coverage", f"{bt['in_50'].mean():.0%}", "target 50%", delta_color="off")
    m[2].metric("90% interval coverage", f"{bt['in_90'].mean():.0%}", "target 90%", delta_color="off")
    m[3].metric("Mean error (bias)", f"{(bt['mean'] - bt['actual']).mean():+.1f} €/MWh",
                help="Positive means the model forecast too high on average.")
    rows = []
    for label, (v, folder, _) in ALL_VERSIONS.items():
        if not (folder / "backtest.csv").exists():
            continue
        b = pd.read_csv(folder / "backtest.csv", dtype={"target": str})
        mo = b["target"].str[4:].astype(int)
        err = b["mean"] - b["actual"]
        rows.append({"model": label + (" (shown)" if v == version else ""),
                     "mean abs. error €/MWh": b["abs_err_mean"].mean(), "bias €/MWh": err.mean(),
                     "bias Jun to Oct €/MWh": err[(mo >= 6) & (mo <= 10)].mean(),
                     "50% coverage": 100 * b["in_50"].mean(), "90% coverage": 100 * b["in_90"].mean(),
                     "CRPS": b["crps"].mean()})
    if len(rows) > 1:
        st.caption("All versions on the same months")
        num = lambda f: st.column_config.NumberColumn(format=f)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"50% coverage": num("%.0f%%"), "90% coverage": num("%.0f%%"),
                                    "mean abs. error €/MWh": num("%.1f"), "bias €/MWh": num("%+.1f"),
                                    "bias Jun to Oct €/MWh": num("%+.1f"), "CRPS": num("%.1f")})
    fig = go.Figure()
    intervals(fig, [mlabel(t) for t in bt["target"]], bt, bt["actual"], "Realised base index")
    fig.add_trace(go.Scatter(x=[mlabel(t) for t in bt["target"]], y=bt["naive_30d"], mode="markers", name="Naive forecast",
                             marker=dict(symbol="line-ew", size=14, line=dict(color=INK2, width=2)),
                             hovertemplate="Naive: %{y:.1f}<extra></extra>"))
    show(fig_defaults(fig, "Forecast at each month end vs. the realised month", "€/MWh", 440),
         bt[["target", "origin", "actual", "mean", "p05", "p95", "naive_30d", "pit", "crps"]].round(2), key="bt")

# ---------------------------------------------------------------- latent states
st.subheader("What the model sees: weekly states")
STATES = {
    "v3_solar": None, "v3_seasonal": None, "v3_both": None,
    "v1": (("level_mean", "Price level, €/MWh"), ("beta_mean", "Merit-order slope, €/MWh per s.d. of residual load"),
           ("sigma_mean", "Daily noise scale, €/MWh")),
    "v2": (("m_mean", "Price as a multiple of gas plant cost"), ("s_mean", "Slope: change in the multiple per s.d. of residual load"),
           ("sigma_mean", "Noise scale (before scaling with fuel cost)")),
}[version] or (("m_mean", "Price as a multiple of gas plant cost"),
                ("s_mean", "Slope: change in the multiple per s.d. of residual load"),
                ("sigma_mean", "Noise scale (before scaling with fuel cost)"))
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[t for _, t in STATES])
for i, (col, name) in enumerate(STATES, 1):
    fig.add_trace(go.Scatter(x=states["week_start"], y=states[col], name=name, mode="lines",
                             line=dict(color=MODEL, width=2), showlegend=False,
                             hovertemplate="%{y:.2f}<extra></extra>"), row=i, col=1)
fig.update_annotations(font=dict(size=12, color=INK2), x=0, xanchor="left")
show(fig_defaults(fig, "", "", 620), states.round(3), key="states")
st.caption({"v3_solar": None, "v3_seasonal": None, "v3_both": None, "v1": "Posterior means. The level stands in for fuel and carbon costs; the slope is how much prices react "
                  "to residual load, which is steep when gas sets the price.",
            "v2": "Posterior means. A multiple near 1 means the power price sits at the cost of gas-fired power; "
                  "lower means cheaper plants (renewables, coal, imports) set the price more often."}[version]
           or "Posterior means. A multiple near 1 means the power price sits at the cost of gas-fired power; "
              "lower means cheaper plants (renewables, coal, imports) set the price more often.")

with st.expander("Model details and parameters"):
    st.markdown((REPORTS / "README.md").read_text().split("## Fit")[0].split("## What the model does")[1])
    p = params["params"]
    st.dataframe(pd.DataFrame([{"parameter": k, **{kk: round(v, 2) for kk, v in d.items()}} for k, d in p.items()]),
                 hide_index=True, width="stretch")
