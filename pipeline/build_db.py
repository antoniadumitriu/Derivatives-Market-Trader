"""Build the local DuckDB warehouse from the CSVs in data/.

  python -m pipeline.build_db            # writes data/warehouse.duckdb (not committed)

Raw tables (copies of the CSVs as of the last build):
  eex_settlements, da_prices, generation, power_forecasts, crossborder,
  installed_capacity, weather_actuals, weather_forecasts, weather_points,
  runs_eex, runs_energy_charts, runs_weather

Daily tables (delivery day in Europe/Berlin), for the dashboard:
  prices_daily, generation_daily, crossborder_daily, power_forecasts_daily, weather_daily

Modeling tables (rebuilt each time):
  fuel_daily      TTF gas, EUA carbon and the marginal cost of a gas-fired plant, per day
  de_daily        one row per German delivery day (Europe/Berlin): day-ahead base
                  and peak price, load, residual load, wind, solar, net imports,
                  temperature, gas, carbon and gas-plant cost of the day before, calendar flags
  de_month_index  realised monthly base average of the day-ahead price, i.e. what
                  a German base month future settles against
  de_base_month_futures  EEX German base month futures settlements per trade date,
                  with months-to-delivery (1 = front month)
"""

import os
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "warehouse.duckdb"

# Points averaged for the German temperature series (demand centres only).
DE_DEMAND_POINTS = ["de_hamburg", "de_berlin", "de_ruhr", "de_frankfurt", "de_munich", "de_stuttgart"]

VIEWS = {
    "eex_settlements": "data/eex/settlements/*.csv",
    "da_prices": "data/energy_charts/prices/*.csv",
    "generation": "data/energy_charts/generation/*.csv",
    "power_forecasts": "data/energy_charts/forecasts/*.csv",
    "crossborder": "data/energy_charts/crossborder/*.csv",
    "installed_capacity": "data/energy_charts/installed_capacity.csv",
    "weather_actuals": "data/weather/actuals/*.csv",
    "weather_forecasts": "data/weather/forecasts/*.csv",
    "weather_points": "data/weather/points.csv",
    "market_prices": "data/market/daily.csv",
    "ttf_curve": "data/market/ttf_curve.csv",
}

# Gas-fired plant used for the marginal cost: efficiency and emissions per MWh of gas burnt.
CCGT_EFFICIENCY = 0.50
GAS_EMISSIONS_T_PER_MWH = 0.202

# A daily value is kept only when every hour of the local day has data for it; otherwise it is
# NULL. A partly published day (e.g. today, or a collection run that failed half way) would
# otherwise show as a full day averaged over the hours that happen to be there.
HOUR = "date_trunc('hour', CAST(timestamp_utc AS TIMESTAMPTZ))"
HOURS_IN_DAY = ("date_diff('hour', timezone('Europe/Berlin', CAST(day AS TIMESTAMP)), "
                "timezone('Europe/Berlin', CAST(day + 1 AS TIMESTAMP)))")


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def complete_avg(expr: str, scale: str = "") -> str:
    """avg(expr) over a local day, or NULL unless expr has a value in every hour of that day.
    Needs columns `day` and `hour_utc` in scope."""
    return (f"CASE WHEN count(DISTINCT CASE WHEN ({expr}) IS NOT NULL THEN hour_utc END) >= {HOURS_IN_DAY} "
            f"THEN avg({expr}){scale} END")


RUN_LOGS = {
    "runs_market": "data/market/runs.csv",
    "runs_eex": "data/eex/runs.csv",
    "runs_energy_charts": "data/energy_charts/runs.csv",
    "runs_weather": "data/weather/runs.csv",
}


def build(path: pathlib.Path = DB) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone = 'UTC'")
    for name, pattern in VIEWS.items():
        con.execute(f"""CREATE OR REPLACE TABLE {name} AS
            SELECT * FROM read_csv('{ROOT / pattern}', union_by_name = true, header = true)""")
    for name, pattern in RUN_LOGS.items():
        con.execute(f"""CREATE OR REPLACE TABLE {name} AS
            SELECT * FROM read_csv('{ROOT / pattern}', header = true, all_varchar = true)""")

    # Daily aggregates (delivery day in Europe/Berlin) for every country, zone and point.
    local_day = "CAST(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ)) AS DATE)"
    con.execute(f"""
    CREATE OR REPLACE TABLE prices_daily AS
    SELECT zone, {local_day} AS day, avg(price_eur_mwh) AS base,
           avg(price_eur_mwh) FILTER (WHERE hour(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ))) BETWEEN 8 AND 19
                                       AND isodow({local_day}) <= 5) AS peak,
           min(price_eur_mwh) AS min, max(price_eur_mwh) AS max,
           count(*) FILTER (WHERE price_eur_mwh < 0) AS negative_intervals, count(*) AS intervals
    FROM da_prices GROUP BY ALL ORDER BY zone, day""")
    for name, src, key in (("generation_daily", "generation", "country"),
                           ("crossborder_daily", "crossborder", "country"),
                           ("power_forecasts_daily", "power_forecasts", "country"),
                           ("weather_daily", "weather_actuals", "point")):
        cols = [c for c in con.execute(f"SELECT * FROM {src} LIMIT 0").fetchdf().columns
                if c not in ("timestamp_utc", key)]
        aggs = ", ".join(f'{complete_avg(f"{qi(c)}")} AS {qi(c)}' for c in cols)
        con.execute(f"""CREATE OR REPLACE TABLE {name} AS
            SELECT {key}, day, {aggs}
            FROM (SELECT *, {local_day} AS day, {HOUR} AS hour_utc FROM {src})
            GROUP BY {key}, day ORDER BY {key}, day""")

    # Daily gas and carbon. EUA: EEX December front settlement where the scraper has it;
    # before that, the WisdomTree Carbon ETC converted to EUR and scaled to the EEX level
    # with the median ratio over the overlap (an approximation; the ETC carries fees and roll).
    con.execute("""
    CREATE OR REPLACE TABLE fuel_daily AS
    WITH m AS (
        SELECT CAST(date AS DATE) AS day,
               max(close) FILTER (WHERE symbol = 'TTF=F') AS ttf,
               max(close) FILTER (WHERE symbol = 'CARB.L') / max(close) FILTER (WHERE symbol = 'EURUSD=X') AS carb_eur
        FROM market_prices GROUP BY ALL
    ), eex AS (
        SELECT CAST(trade_date AS DATE) AS day, arg_min(settlement, maturity) AS eua_eex
        FROM eex_settlements
        WHERE short_code = 'FEUA' AND settlement IS NOT NULL AND CAST(maturity AS VARCHAR) LIKE '%12'
          AND maturity >= year(CAST(trade_date AS DATE)) * 100 + month(CAST(trade_date AS DATE))
        GROUP BY ALL
    ), ratio AS (
        SELECT median(eua_eex / carb_eur) AS r FROM m JOIN eex USING (day) WHERE carb_eur > 0
    ), cal AS (
        SELECT CAST(unnest(generate_series(DATE '2020-01-01', current_date, INTERVAL 1 DAY)) AS DATE) AS day
    ), j AS (
        SELECT cal.day, m.ttf, eex.eua_eex,
               -- the proxy only fills the years before the EEX history; afterwards a missing
               -- EEX day carries the last EEX settlement forward
               CASE WHEN cal.day < (SELECT min(day) FROM eex) THEN m.carb_eur * (SELECT r FROM ratio) END AS eua_proxy
        FROM cal LEFT JOIN m USING (day) LEFT JOIN eex USING (day)
    )
    SELECT day,
           last_value(ttf IGNORE NULLS) OVER w AS ttf_eur_mwh,
           last_value(coalesce(eua_eex, eua_proxy) IGNORE NULLS) OVER w AS eua_eur_t,
           last_value(CASE WHEN eua_eex IS NOT NULL THEN 'eex' WHEN eua_proxy IS NOT NULL THEN 'proxy' END IGNORE NULLS) OVER w AS eua_source
    FROM j WINDOW w AS (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ORDER BY day
    """)
    con.execute(f"""
    ALTER TABLE fuel_daily ADD COLUMN gas_srmc_eur_mwh DOUBLE;
    UPDATE fuel_daily SET gas_srmc_eur_mwh = (ttf_eur_mwh + {GAS_EMISSIONS_T_PER_MWH} * eua_eur_t) / {CCGT_EFFICIENCY};
    """)

    points = ", ".join(f"'{p}'" for p in DE_DEMAND_POINTS)
    con.execute(f"""
    CREATE OR REPLACE TABLE de_daily AS
    WITH price AS (
        SELECT CAST(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ)) AS DATE) AS day,
               price_eur_mwh,
               hour(timezone('Europe/Berlin', CAST(timestamp_utc AS TIMESTAMPTZ))) AS h
        FROM da_prices WHERE zone = 'DE-LU'
    ), p AS (
        SELECT day,
               avg(price_eur_mwh) AS base_price,
               avg(price_eur_mwh) FILTER (WHERE h BETWEEN 8 AND 19 AND isodow(day) <= 5) AS peak_price,
               min(price_eur_mwh) AS min_price,
               max(price_eur_mwh) AS max_price,
               count(*) AS price_intervals
        FROM price GROUP BY day
    ), g AS (
        SELECT day,
               {complete_avg("load", " / 1000")} AS load_gw,
               {complete_avg("residual_load", " / 1000")} AS residual_load_gw,
               {complete_avg("wind_onshore + coalesce(wind_offshore, 0)", " / 1000")} AS wind_gw,
               {complete_avg("solar", " / 1000")} AS solar_gw,
               {complete_avg("fossil_gas", " / 1000")} AS gas_gen_gw,
               {complete_avg("fossil_brown_coal_lignite + coalesce(fossil_hard_coal, 0)", " / 1000")} AS coal_gen_gw,
               count(DISTINCT hour_utc) AS gen_hours
        FROM (SELECT *, {local_day} AS day, {HOUR} AS hour_utc FROM generation WHERE country = 'de') GROUP BY day
    ), x AS (
        SELECT day, {complete_avg("sum")} AS net_imports_gw
        FROM (SELECT *, {local_day} AS day, {HOUR} AS hour_utc FROM crossborder WHERE country = 'de') GROUP BY day
    ), w AS (
        -- all six demand centres, each with a complete day
        SELECT day, avg(temperature_2m) AS temp_c
        FROM weather_daily WHERE point IN ({points}) GROUP BY day
        HAVING count(temperature_2m) = {len(DE_DEMAND_POINTS)}
    )
    SELECT p.day, p.base_price, p.peak_price, p.min_price, p.max_price, p.price_intervals,
           g.load_gw, g.residual_load_gw, g.wind_gw, g.solar_gw, g.gas_gen_gw, g.coal_gen_gw, g.gen_hours,
           x.net_imports_gw, w.temp_c,
           f.ttf_eur_mwh, f.eua_eur_t, f.gas_srmc_eur_mwh,
           isodow(p.day) >= 6 AS is_weekend,
           strftime(p.day, '%Y%m') AS month
    FROM p LEFT JOIN g USING (day) LEFT JOIN x USING (day) LEFT JOIN w USING (day)
    LEFT JOIN fuel_daily f ON f.day = p.day - INTERVAL 1 DAY  -- fuel prices known at auction time
    ORDER BY p.day
    """)

    con.execute("""
    CREATE OR REPLACE TABLE de_month_index AS
    SELECT month, avg(base_price) AS base_index, avg(peak_price) AS peak_index,
           count(*) AS days, max(day) AS last_day,
           count(*) = day(last_day(min(day))) AS complete
    FROM de_daily GROUP BY month ORDER BY month
    """)

    con.execute("""
    CREATE OR REPLACE TABLE de_base_month_futures AS
    SELECT CAST(trade_date AS DATE) AS trade_date, CAST(maturity AS VARCHAR) AS month,
           settlement, volume, gross_open_interest_lots AS open_interest_lots,
           (CAST(substr(CAST(maturity AS VARCHAR), 1, 4) AS INT) * 12 + CAST(substr(CAST(maturity AS VARCHAR), 5, 2) AS INT))
           - (year(CAST(trade_date AS DATE)) * 12 + month(CAST(trade_date AS DATE))) AS months_ahead
    FROM eex_settlements
    WHERE short_code = 'DEBM' AND settlement IS NOT NULL
    ORDER BY trade_date, month
    """)
    return con


def rebuild() -> None:
    """Build into a temporary file and swap it in, so readers that have the warehouse
    open (the dashboard, a running model) keep working."""
    tmp = DB.with_name("warehouse.building.duckdb")
    tmp.unlink(missing_ok=True)
    build(tmp).close()
    os.replace(tmp, DB)


if __name__ == "__main__":
    rebuild()
    con = duckdb.connect(str(DB), read_only=True)
    for t in ("de_daily", "de_month_index", "de_base_month_futures"):
        n, lo, hi = con.execute(f"SELECT count(*), min({'day' if t == 'de_daily' else 'month' if t == 'de_month_index' else 'trade_date'}), "
                                f"max({'day' if t == 'de_daily' else 'month' if t == 'de_month_index' else 'trade_date'}) FROM {t}").fetchone()
        print(f"{t}: {n} rows, {lo} to {hi}")
    print(f"Wrote {DB}")
