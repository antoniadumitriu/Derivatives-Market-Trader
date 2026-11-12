"""Shared helpers for the dashboard: warehouse connection, chart styling, small widgets."""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.build_db import DB, rebuild  # noqa: E402

REPORTS = ROOT / "reports" / "de_base_month_v3_both"  # current model
REPORTS_V1 = ROOT / "reports" / "de_base_month"

# ---------------------------------------------------------------- palette
# Categorical hues, assigned in this fixed order and never cycled (max 8 series).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
MODEL, OBSERVED = SERIES[0], SERIES[1]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}

pio.templates["dmt"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", color=INK2, size=12),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
    xaxis=dict(showgrid=False, linecolor=INK2, ticks="", zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False, ticks=""),
    # Legend under the plot, so a legend that wraps never runs into the title.
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
))
pio.templates.default = "dmt"


def color_map(items) -> dict:
    """Stable entity -> colour mapping in fixed order (colour follows the entity)."""
    return {item: SERIES[i % len(SERIES)] for i, item in enumerate(items)}


# ---------------------------------------------------------------- data

def page(title: str, caption: str | None = None):
    st.title(title)
    if caption:
        st.caption(caption)


@st.cache_resource(show_spinner="Building the warehouse from data/ ...")
def _connection(stamp: float) -> duckdb.DuckDBPyConnection:
    if not DB.exists() or DB.stat().st_mtime < stamp:
        rebuild()
    try:
        return duckdb.connect(str(DB), read_only=True)
    except duckdb.IOException:
        # Usually a warehouse written by a different DuckDB version: rebuild it with this one.
        rebuild()
        return duckdb.connect(str(DB), read_only=True)



def data_stamp() -> float:
    """Latest modification time of any data CSV, so the warehouse rebuilds after a git pull."""
    return max(p.stat().st_mtime for p in (ROOT / "data").rglob("*.csv"))


def con() -> duckdb.DuckDBPyConnection:
    return _connection(max(data_stamp(), DB.stat().st_mtime if DB.exists() else 0))


@st.cache_data(show_spinner=False, max_entries=500)
def _q(stamp: float, sql: str, params: tuple) -> pd.DataFrame:
    return _connection(stamp).execute(sql, list(params)).df()


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a query; results are cached until the data files or the warehouse change."""
    stamp = max(data_stamp(), DB.stat().st_mtime if DB.exists() else 0)
    return _q(stamp, sql, tuple(params))


def rebuild_button():
    with st.sidebar:
        st.caption("The warehouse rebuilds automatically when the CSVs change (for example after `git pull`).")
        if st.button("Rebuild warehouse now"):
            rebuild()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()


def date_range(key: str, default_days: int, min_date: dt.date, max_date: dt.date):
    presets = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "3Y": 1096, "All": None}
    labels = list(presets)
    default = min(labels[:-1], key=lambda k: abs(presets[k] - default_days))
    choice = st.segmented_control("Range", labels, default=default, key=f"{key}_preset")
    days = presets.get(choice or default)
    start = min_date if days is None else max(min_date, max_date - dt.timedelta(days=days))
    return start, max_date


def fig_defaults(fig: go.Figure, title: str, ytitle: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, height=height, yaxis_title=ytitle)
    if fig.layout.xaxis.title.text:  # make room for the x-axis title above the legend
        fig.update_layout(legend_y=-0.22)
    return fig


def show(fig: go.Figure, table: pd.DataFrame | None = None, key: str | None = None):
    st.plotly_chart(fig, key=key)  # fills the container by default
    if table is not None:
        with st.expander("Table view"):
            st.dataframe(table, width="stretch", hide_index=True)
