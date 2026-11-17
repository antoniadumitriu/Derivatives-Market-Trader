"""Model v2: German base month index, with gas and carbon costs.

Daily German day-ahead base price y_d (EUR/MWh):

    c_d  = marginal cost of a gas-fired plant on the day before delivery:
           (TTF gas + 0.202 t/MWh x EUA carbon) / 0.50 efficiency, in hundreds of EUR/MWh
    y_d ~ StudentT(nu, c_d * (m_w + s_w * z_d) + gamma * weekend_d,  sigma_w * c_d ** phi)

    m_w   weekly random walk: price as a multiple of the gas-plant cost at average residual load
    s_w   weekly random walk: how much that multiple moves per s.d. of residual load
    sigma_w  weekly random walk (log): volatility; phi in [0, 1] sets how much of the
             volatility scales with fuel costs

Because price is tied to fuel cost, a jump in gas moves the level and the merit-order
slope together, which v1 had to learn week by week through its free-floating level.

Forecast of a delivery month at an origin date:
  * fuel cost for the month from the TTF futures price for that month (EEX settlement,
    else Yahoo; the TTF front
    month in the backtest, where old contract prices are not available) and the EUA
    December future, with lognormal gas-price uncertainty growing with time to delivery
    (volatility estimated from the last 60 days of TTF);
  * m, s and sigma carried forward along their random walks;
  * residual load drawn from the same calendar month of the last 3 years.

Usage:
  python -m models.de_base_month_v2                  # fit, forecast the next 3 months
  python -m models.de_base_month_v2 --backtest 24
"""

import argparse
import calendar
import datetime as dt
import json
import pathlib
import time

import duckdb
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_value

from models.de_base_month import crps, next_month, summary
from connectors.common import DATA
from pipeline.build_db import CCGT_EFFICIENCY, DB, GAS_EMISSIONS_T_PER_MWH, rebuild

numpyro.set_host_device_count(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "de_base_month_v2"
CLIMATE_YEARS = 3
SCALE = 100.0  # fuel cost in hundreds of EUR/MWh keeps the multiples near 1


def srmc(ttf, eua):
    return (np.asarray(ttf) + GAS_EMISSIONS_T_PER_MWH * np.asarray(eua)) / CCGT_EFFICIENCY


# ---------------------------------------------------------------- data

def con():
    return duckdb.connect(str(DB), read_only=True)


def load_daily() -> pd.DataFrame:
    df = con().execute("""SELECT day, base_price, residual_load_gw, load_gw, wind_gw, solar_gw, is_weekend,
                                 gas_srmc_eur_mwh AS cost
                          FROM de_daily
                          WHERE gen_hours >= 23 AND residual_load_gw IS NOT NULL AND gas_srmc_eur_mwh IS NOT NULL
                          ORDER BY day""").df()
    df["day"] = pd.to_datetime(df["day"])
    return df


def load_fuel() -> pd.DataFrame:
    f = con().execute("SELECT day, ttf_eur_mwh AS ttf, eua_eur_t AS eua FROM fuel_daily ORDER BY day").df()
    f["day"] = pd.to_datetime(f["day"])
    return f.set_index("day")


def load_capacity() -> pd.DataFrame:
    """German installed wind (onshore + offshore) and solar capacity by month, GW."""
    c = con().execute("""
        SELECT period AS month,
               sum(capacity_gw) FILTER (WHERE technology IN ('wind_onshore', 'wind_offshore')) AS wind,
               sum(capacity_gw) FILTER (WHERE technology = 'solar_ac') AS solar
        FROM installed_capacity WHERE country = 'de' AND time_step = 'monthly' GROUP BY 1 ORDER BY 1""").df()
    return c.set_index("month")


def load_curve() -> pd.DataFrame:
    c = con().execute("SELECT CAST(trade_date AS DATE) AS trade_date, CAST(delivery_month AS VARCHAR) AS month, close FROM ttf_curve").df()
    c["trade_date"] = pd.to_datetime(c["trade_date"])
    return c


# ---------------------------------------------------------------- model

def model(week, z, weekend, cost, n_weeks, month=None, y=None):
    m0 = numpyro.sample("m0", dist.Normal(1.0, 0.5))
    s0 = numpyro.sample("s0", dist.Normal(0.2, 0.2))
    logsig0 = numpyro.sample("logsig0", dist.Normal(jnp.log(15.0), 1.0))
    tau_m = numpyro.sample("tau_m", dist.HalfNormal(0.1))
    tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.05))
    tau_logsig = numpyro.sample("tau_logsig", dist.HalfNormal(0.3))
    phi = numpyro.sample("phi", dist.Beta(2.0, 2.0))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 30.0))
    # Heavy tails, but at least 2.1 degrees of freedom so the variance stays finite.
    nu = numpyro.deterministic("nu", 2.1 + numpyro.sample("nu_raw", dist.Gamma(2.0, 0.15)))

    with numpyro.plate("steps", n_weeks - 1):
        e_m = numpyro.sample("e_m", dist.Normal(0.0, 1.0))
        e_s = numpyro.sample("e_s", dist.Normal(0.0, 1.0))
        e_logsig = numpyro.sample("e_logsig", dist.Normal(0.0, 1.0))

    def walk(x0, tau, e):
        return jnp.concatenate([jnp.array([x0]), x0 + jnp.cumsum(tau * e)])

    m = numpyro.deterministic("m", walk(m0, tau_m, e_m))
    s = numpyro.deterministic("s", walk(s0, tau_s, e_s))
    logsig = numpyro.deterministic("logsig", walk(logsig0, tau_logsig, e_logsig))

    mult = m[week]
    if month is not None:
        # Version 3: a shift in the multiple for each calendar month, centred so the 12 shifts
        # average to zero and the random walk keeps the underlying level.
        raw = numpyro.sample("season_raw", dist.Normal(0.0, 0.2).expand([12]))
        season = numpyro.deterministic("season", raw - raw.mean())
        mult = mult + season[month]
    mu = SCALE * cost * (mult + s[week] * z) + gamma * weekend
    scale = jnp.exp(logsig[week]) * cost ** phi
    with numpyro.plate("days", len(z)):
        numpyro.sample("y", dist.StudentT(nu, mu, scale), obs=y)


class Fit:
    def __init__(self, daily, origin, samples=500, warmup=500, chains=2, seed=0, seasonal=False, capacity=None):
        self.seasonal = seasonal
        # capacity: monthly installed wind/solar (load_capacity()). When given, residual-load
        # scenarios rescale past years' solar output to the solar capacity at the origin.
        self.capacity = capacity
        d = daily[daily["day"] <= origin].copy()
        self.origin, self.day0, self.history = origin, d["day"].min(), d
        self.z_mean, self.z_sd = d["residual_load_gw"].mean(), d["residual_load_gw"].std()
        week = ((d["day"] - self.day0).dt.days // 7).to_numpy()
        self.n_weeks = int(week.max()) + 1
        t0 = time.time()
        # Start every chain from plausible values: random starts on the scale parameters can
        # leave a chain stuck in a region where the random walks explode.
        init = init_to_value(values={"m0": 1.0, "s0": 0.2, "logsig0": float(np.log(15.0)), "tau_m": 0.03,
                                     "tau_s": 0.02, "tau_logsig": 0.1, "phi": 0.5, "gamma": 0.0, "nu_raw": 3.0,
                                     **({"season_raw": np.zeros(12)} if seasonal else {})})
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=init), num_warmup=warmup, num_samples=samples,
                    num_chains=chains, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), week=jnp.array(week), z=jnp.array(self.zscore(d["residual_load_gw"])),
                 weekend=jnp.array(d["is_weekend"].astype(float).to_numpy()),
                 cost=jnp.array(d["cost"].to_numpy() / SCALE), n_weeks=self.n_weeks,
                 month=jnp.array(d["day"].dt.month.to_numpy() - 1) if seasonal else None,
                 y=jnp.array(d["base_price"].to_numpy()))
        self.seconds = time.time() - t0
        self.post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        self.mcmc = mcmc

    def zscore(self, x):
        return (np.asarray(x) - self.z_mean) / self.z_sd

    def residual_load_scenarios(self, month):
        y, m = int(month[:4]), int(month[4:])
        n = calendar.monthrange(y, m)[1]
        h, paths = self.history, []
        cap = self.capacity
        if cap is not None:
            now = cap.loc[:self.origin.strftime("%Y-%m")].iloc[-1]
        for year in sorted(h["day"].dt.year.unique(), reverse=True):
            rows = h[(h["day"].dt.year == year) & (h["day"].dt.month == m)]
            if cap is not None and f"{year}-{m:02d}" in cap.index:
                then = cap.loc[f"{year}-{m:02d}"]
                # Same weather, today's solar fleet: solar output scales with installed capacity.
                # Wind is left as it was: in backtests, scaling it too overstated winter wind
                # (new wind capacity does not add output one-for-one, e.g. curtailment).
                part = (rows["load_gw"] - rows["wind_gw"]
                        - rows["solar_gw"] * now["solar"] / then["solar"]).to_numpy()
            else:
                part = rows["residual_load_gw"].to_numpy()
            if len(part) >= 28 and year < y:
                paths.append(np.resize(part, n))
            if len(paths) == CLIMATE_YEARS:
                break
        return paths

    def forecast_month(self, month, gas_forward, eua, gas_vol, seed=1):
        """gas_forward: TTF price for the delivery month at the origin; eua: EUA price;
        gas_vol: annualised TTF volatility. Returns draws of the month's base index."""
        rng = np.random.default_rng(seed)
        y, mo = int(month[:4]), int(month[4:])
        days = pd.date_range(dt.date(y, mo, 1), periods=calendar.monthrange(y, mo)[1], freq="D")
        weekend = (days.dayofweek >= 5).astype(float)
        last = self.n_weeks - 1
        steps = np.maximum((((days - self.day0).days // 7) - last).to_numpy(), 1)
        p, S, H = self.post, len(self.post["nu"]), int(steps.max())

        def carry(x_last, tau):
            return x_last[:, None] + np.cumsum(tau[:, None] * rng.standard_normal((S, H)), axis=1)

        m = carry(p["m"][:, last], p["tau_m"])[:, steps - 1]
        if self.seasonal:
            m = m + p["season"][:, mo - 1][:, None]
        s = carry(p["s"][:, last], p["tau_s"])[:, steps - 1]
        sig = np.exp(carry(p["logsig"][:, last], p["tau_logsig"])[:, steps - 1])
        # Gas price during delivery: forward with lognormal uncertainty up to mid-delivery.
        T = max((days[len(days) // 2] - self.origin).days, 1) / 365.0
        shock = np.exp(gas_vol * np.sqrt(T) * rng.standard_normal(S) - 0.5 * gas_vol**2 * T)
        cost = srmc(gas_forward * shock, eua)[:, None] / SCALE
        paths = self.residual_load_scenarios(month)
        z = np.stack([self.zscore(paths[i]) for i in rng.integers(len(paths), size=S)])
        mu = SCALE * cost * (m + s * z) + p["gamma"][:, None] * weekend[None, :]
        noise = rng.standard_t(p["nu"][:, None], size=mu.shape) * sig * cost ** p["phi"][:, None]
        return (mu + noise).mean(axis=1)


def gas_vol_at(fuel: pd.DataFrame, origin) -> float:
    ttf = fuel.loc[:origin, "ttf"].dropna()
    r = np.log(ttf[ttf.diff() != 0]).diff().dropna().iloc[-60:]  # trading days only
    return float(r.std() * np.sqrt(252))


def forward_at(fuel, curve, origin, month, backtest=False):
    """TTF price for `month` as known at `origin`, and where it came from."""
    if not backtest:
        e = con().execute("""SELECT settlement FROM eex_settlements
                             WHERE commodity = 'NATGAS' AND area = 'TTF' AND product = 'Physical'
                               AND maturity_type = 'Month' AND CAST(maturity AS VARCHAR) = ?
                               AND settlement IS NOT NULL AND CAST(trade_date AS DATE) <= ?
                             ORDER BY trade_date DESC LIMIT 1""", [month, origin.date()]).fetchone()
        if e:
            return float(e[0]), "EEX TTF settlement"
        c = curve[(curve["month"] == month) & (curve["trade_date"] <= origin)]
        if len(c):
            return float(c.sort_values("trade_date")["close"].iloc[-1]), "TTF contract"
    # Front month averaged over the last 5 trading days, so a roll in the continuous series
    # around month end does not dominate.
    ttf = fuel.loc[:origin, "ttf"].dropna()
    return float(ttf[ttf.diff() != 0].iloc[-5:].mean()), "TTF front month"


# ---------------------------------------------------------------- runs

def append_log(res: pd.DataFrame, model_name: str, origin) -> None:
    """Add this run's forecasts to the forecast log in the data repo, one row per delivery
    month, so the model can later be scored against futures out of sample."""
    log = DATA / "model" / "forecast_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = res.reset_index().assign(
        run_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), model=model_name,
        data_through=str(pd.Timestamp(origin).date()))
    rows["months_ahead"] = [
        (int(m[:4]) * 12 + int(m[4:])) - (pd.Timestamp(origin).year * 12 + pd.Timestamp(origin).month) for m in rows["month"]]
    cols = ["run_utc", "model", "data_through", "month", "months_ahead", "mean", "p05", "p25", "p50", "p75", "p95",
            "futures", "futures_trade_date", "p_futures_above_model", "ttf_forward", "ttf_source", "eua",
            "gas_plant_cost"]
    rows[cols].to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"Appended {len(rows)} rows to {log}")


def run_forecast(daily, fuel, curve, seasonal=False, out=OUT, capacity=None, live=False):
    origin = daily["day"].max()
    fit = Fit(daily, origin, samples=1000, warmup=1000, seasonal=seasonal, capacity=capacity)
    print(f"Fit through {origin.date()}: {fit.n_weeks} weeks, {fit.seconds:.0f}s, {fit.divergences} divergences")
    fut = con().execute("SELECT trade_date, month, settlement FROM de_base_month_futures").df()
    last_trade = fut["trade_date"].max()
    eex = fut[fut["trade_date"] == last_trade].set_index("month")["settlement"]
    eua = float(fuel.loc[:origin, "eua"].dropna().iloc[-1])
    vol = gas_vol_at(fuel, origin)
    rows = []
    for k in (1, 2, 3):
        month = next_month(origin.strftime("%Y%m"), k)
        fwd, src = forward_at(fuel, curve, pd.Timestamp(last_trade), month)
        draws = fit.forecast_month(month, fwd, eua, vol)
        s = summary(draws)
        f = eex.get(month)
        s.update({"month": month, "futures": f, "futures_trade_date": str(pd.Timestamp(last_trade).date()),
                  "p_futures_above_model": float((draws < f).mean()) if f is not None else None,
                  "ttf_forward": fwd, "ttf_source": src, "eua": eua, "gas_vol": vol,
                  "gas_plant_cost": float(srmc(fwd, eua))})
        rows.append(s)
    res = pd.DataFrame(rows).set_index("month")
    if live:
        # Daily runs: outputs go to the data repo (they contain futures prices), plus the log.
        append_log(res, out.name, origin)
        out = DATA / "model" / out.name
        out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "forecast.csv")
    p = fit.post
    params = {k: summary(p[k]) for k in ("tau_m", "tau_s", "tau_logsig", "phi", "gamma", "nu")}
    params["m_last_week"] = summary(p["m"][:, -1])
    params["s_last_week"] = summary(p["s"][:, -1])
    if seasonal:
        for i, name in enumerate(calendar.month_abbr[1:]):
            params[f"season_{name.lower()}"] = summary(p["season"][:, i])
    (out / "params.json").write_text(json.dumps({"origin": str(origin.date()), "weeks": fit.n_weeks,
                                                 "divergences": fit.divergences, "seconds": round(fit.seconds),
                                                 "params": params}, indent=2, default=float))
    pd.DataFrame({"week_start": fit.day0 + pd.to_timedelta(np.arange(fit.n_weeks) * 7, "D"),
                  "m_mean": p["m"].mean(0), "s_mean": p["s"].mean(0),
                  "sigma_mean": np.exp(p["logsig"]).mean(0)}).to_csv(out / "states.csv", index=False)
    print(res[["mean", "p05", "p95", "futures", "p_futures_above_model", "ttf_forward", "ttf_source"]].round(2).to_string())


def run_backtest(daily, fuel, months_back, seasonal=False, out=OUT, capacity=None):
    index = con().execute("SELECT month, base_index, complete FROM de_month_index").df().set_index("month")
    targets = index[index["complete"]].index.tolist()[-months_back:]
    rows = []
    for target in targets:
        prev = next_month(target, -1)
        py, pm = int(prev[:4]), int(prev[4:])
        origin = pd.Timestamp(dt.date(py, pm, calendar.monthrange(py, pm)[1]))
        fit = Fit(daily, origin, samples=400, warmup=400, seasonal=seasonal, capacity=capacity)
        fwd, src = forward_at(fuel, None, origin, target, backtest=True)
        eua = float(fuel.loc[:origin, "eua"].dropna().iloc[-1])
        draws = fit.forecast_month(target, fwd, eua, gas_vol_at(fuel, origin))
        actual = float(index.loc[target, "base_index"])
        naive = float(daily[(daily["day"] > origin - pd.Timedelta(days=30)) & (daily["day"] <= origin)]["base_price"].mean())
        s = summary(draws)
        s.update({"target": target, "origin": str(origin.date()), "actual": actual, "naive_30d": naive,
                  "crps": crps(draws, actual), "abs_err_mean": abs(s["mean"] - actual),
                  "abs_err_naive": abs(naive - actual), "in_50": s["p25"] <= actual <= s["p75"],
                  "in_90": s["p05"] <= actual <= s["p95"], "pit": float((draws < actual).mean()),
                  "ttf_forward": fwd, "divergences": fit.divergences, "seconds": round(fit.seconds)})
        rows.append(s)
        print(f"{target}: actual {actual:6.1f}  model {s['mean']:6.1f} [{s['p05']:6.1f}, {s['p95']:6.1f}]  "
              f"naive {naive:6.1f}  pit {s['pit']:.2f}  ({fit.seconds:.0f}s, {fit.divergences} div)", flush=True)
    bt = pd.DataFrame(rows).set_index("target")
    bt.to_csv(out / "backtest.csv")
    print(f"MAE model {bt.abs_err_mean.mean():.1f}, naive {bt.abs_err_naive.mean():.1f}; "
          f"50% coverage {bt.in_50.mean():.0%}, 90% coverage {bt.in_90.mean():.0%}; CRPS {bt.crps.mean():.1f}; "
          f"bias {(bt['mean'] - bt.actual).mean():+.1f}")


def main(seasonal=False, out=OUT, capacity_adjusted=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, default=0)
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="write the forecast to the data repo (model/<version>/) and append to its forecast log")
    args = ap.parse_args()
    out.mkdir(parents=True, exist_ok=True)
    rebuild()
    daily, fuel, curve = load_daily(), load_fuel(), load_curve()
    cap = load_capacity() if capacity_adjusted else None
    if not args.no_forecast:
        run_forecast(daily, fuel, curve, seasonal, out, cap, live=args.live)
    if args.backtest:
        run_backtest(daily, fuel, args.backtest, seasonal, out, cap)


if __name__ == "__main__":
    main()
