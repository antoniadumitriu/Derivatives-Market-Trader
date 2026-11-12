import json
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import INK, INK2, MODEL, OBSERVED, REPORTS, ROOT, fig_defaults, page, rebuild_button, show

page("Model: German base month", "Bayesian model of the monthly German base price index, compared with EEX futures.")
rebuild_button()

if not (REPORTS / "forecast.csv").exists():
    st.warning("No model outputs yet. Run `python -m models.de_base_month --backtest 24` first.")
    st.stop()

fc = pd.read_csv(REPORTS / "forecast.csv", dtype={"month": str})
params = json.loads((REPORTS / "params.json").read_text())
states = pd.read_csv(REPORTS / "states.csv", parse_dates=["week_start"])
bt = pd.read_csv(REPORTS / "backtest.csv", dtype={"target": str}) if (REPORTS / "backtest.csv").exists() else None


def mlabel(m) -> str:
    return pd.Timestamp(f"{m[:4]}-{m[4:6]}-01").strftime("%b %y")


with st.sidebar:
    st.markdown("**Refit the model**")
    st.caption("Refits on the latest data and refreshes the forecast (about 3 minutes). "
               "The 24-month backtest takes about 20 minutes and runs from the command line.")
    if st.button("Refit now"):
        with st.status("Fitting the model ...", expanded=True) as status:
            for cmd in ([sys.executable, "-m", "models.de_base_month"],
                        [sys.executable, "-m", "models.report_de_base_month"]):
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode:
                    st.code(r.stderr[-2000:])
                    status.update(label="Refit failed", state="error")
                    st.stop()
            status.update(label="Refit done", state="complete")
        st.rerun()

st.caption(f"Fitted on data through {params['origin']} ({params['weeks']} weeks, "
           f"{params['divergences']} sampler divergences). EEX settlements from {fc['futures_trade_date'].iloc[0]}.")


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
st.caption("Probability below futures near 50% means model and market agree. Well above 50% means the model sees "
           "futures as expensive, well below means cheap. The model does not see gas prices yet.")

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
    fig = go.Figure()
    intervals(fig, [mlabel(t) for t in bt["target"]], bt, bt["actual"], "Realised base index")
    fig.add_trace(go.Scatter(x=[mlabel(t) for t in bt["target"]], y=bt["naive_30d"], mode="markers", name="Naive forecast",
                             marker=dict(symbol="line-ew", size=14, line=dict(color=INK2, width=2)),
                             hovertemplate="Naive: %{y:.1f}<extra></extra>"))
    show(fig_defaults(fig, "Forecast at each month end vs. the realised month", "€/MWh", 440),
         bt[["target", "origin", "actual", "mean", "p05", "p95", "naive_30d", "pit", "crps"]].round(2), key="bt")

# ---------------------------------------------------------------- latent states
st.subheader("What the model sees: weekly states")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Price level, €/MWh", "Merit-order slope, €/MWh per s.d. of residual load",
                                    "Daily noise scale, €/MWh"))
for i, (col, name) in enumerate((("level_mean", "Level"), ("beta_mean", "Slope"), ("sigma_mean", "Noise scale")), 1):
    fig.add_trace(go.Scatter(x=states["week_start"], y=states[col], name=name, mode="lines",
                             line=dict(color=MODEL, width=2), showlegend=False,
                             hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"), row=i, col=1)
fig.update_annotations(font=dict(size=12, color=INK2), x=0, xanchor="left")
show(fig_defaults(fig, "", "", 620), states.round(2), key="states")
st.caption("Posterior means. The level stands in for fuel and carbon costs; the slope is how much prices react "
           "to residual load, which is steep when gas sets the price.")

with st.expander("Model details and parameters"):
    st.markdown((REPORTS / "README.md").read_text().split("## Fit")[0].split("## What the model does")[1])
    p = params["params"]
    st.dataframe(pd.DataFrame([{"parameter": k, **{kk: round(v, 2) for kk, v in d.items()}} for k, d in p.items()]),
                 hide_index=True, width="stretch")
