"""First Bayesian model: German base month index (what a DE base month future settles to).

Model (daily, German day-ahead base price y_d, EUR/MWh):

    y_d ~ StudentT(nu, level_w + beta_w * z_d + gamma * weekend_d, sigma_w)

    w        week of day d
    z_d      residual load (load minus wind and solar), standardised
    level_w  weekly random walk: the price level set by fuel and carbon costs,
             which this first version does not observe directly
    beta_w   weekly random walk: merit-order slope, EUR/MWh per s.d. of residual load
             (steep when gas is expensive, flat when it is cheap)
    sigma_w  weekly random walk in log space: volatility regime
    nu       Student-t degrees of freedom, for price spikes

Forecast of a delivery month made at an origin date:
  * level, beta and log sigma are carried forward along their random walks;
  * residual load for each delivery day is drawn from the same calendar month of
    recent years (a whole year at a time, so weather spells stay intact);
  * daily prices are simulated and averaged into the month index.

Usage:
  python -m models.de_base_month                     # fit on all data, forecast the next months,
                                                     # compare with EEX futures
  python -m models.de_base_month --backtest 12       # rolling backtest over the last 12 months
"""

import argparse
import calendar
import datetime as dt
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from pipeline.build_db import DB, build

numpyro.set_host_device_count(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "de_base_month"
CLIMATE_YEARS = 3  # residual load scenarios come from this many most recent complete years


# ---------------------------------------------------------------- data

def load_daily() -> pd.DataFrame:
    con = build(DB) if not DB.exists() else __import__("duckdb").connect(str(DB), read_only=True)
    df = con.execute("""SELECT day, base_price, residual_load_gw, is_weekend, month
                        FROM de_daily WHERE gen_hours >= 23 AND residual_load_gw IS NOT NULL
                        ORDER BY day""").df()
    df["day"] = pd.to_datetime(df["day"])
    return df


def load_index() -> pd.DataFrame:
    con = __import__("duckdb").connect(str(DB), read_only=True)
    return con.execute("SELECT month, base_index, complete FROM de_month_index").df()


def load_futures() -> pd.DataFrame:
    con = __import__("duckdb").connect(str(DB), read_only=True)
    df = con.execute("SELECT trade_date, month, settlement, months_ahead FROM de_base_month_futures").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# ---------------------------------------------------------------- model

def model(week, z, weekend, n_weeks, y=None):
    level0 = numpyro.sample("level0", dist.Normal(100.0, 100.0))
    beta0 = numpyro.sample("beta0", dist.Normal(20.0, 20.0))
    logsig0 = numpyro.sample("logsig0", dist.Normal(jnp.log(15.0), 1.0))
    tau_level = numpyro.sample("tau_level", dist.HalfNormal(15.0))
    tau_beta = numpyro.sample("tau_beta", dist.HalfNormal(5.0))
    tau_logsig = numpyro.sample("tau_logsig", dist.HalfNormal(0.3))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 30.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    with numpyro.plate("steps", n_weeks - 1):
        e_level = numpyro.sample("e_level", dist.Normal(0.0, 1.0))
        e_beta = numpyro.sample("e_beta", dist.Normal(0.0, 1.0))
        e_logsig = numpyro.sample("e_logsig", dist.Normal(0.0, 1.0))

    def walk(x0, tau, e):
        return jnp.concatenate([jnp.array([x0]), x0 + jnp.cumsum(tau * e)])

    level = numpyro.deterministic("level", walk(level0, tau_level, e_level))
    beta = numpyro.deterministic("beta", walk(beta0, tau_beta, e_beta))
    logsig = numpyro.deterministic("logsig", walk(logsig0, tau_logsig, e_logsig))

    mu = level[week] + beta[week] * z + gamma * weekend
    with numpyro.plate("days", len(z)):
        numpyro.sample("y", dist.StudentT(nu, mu, jnp.exp(logsig[week])), obs=y)


class Fit:
    def __init__(self, daily: pd.DataFrame, origin: pd.Timestamp, samples=500, warmup=500, chains=2, seed=0):
        d = daily[daily["day"] <= origin].copy()
        self.origin = origin
        self.day0 = d["day"].min()
        self.z_mean, self.z_sd = d["residual_load_gw"].mean(), d["residual_load_gw"].std()
        week = ((d["day"] - self.day0).dt.days // 7).to_numpy()
        self.n_weeks = int(week.max()) + 1
        self.history = d
        t0 = time.time()
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                    num_chains=chains, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), week=jnp.array(week),
                 z=jnp.array(self.zscore(d["residual_load_gw"])),
                 weekend=jnp.array(d["is_weekend"].astype(float).to_numpy()),
                 n_weeks=self.n_weeks, y=jnp.array(d["base_price"].to_numpy()))
        self.seconds = time.time() - t0
        self.post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        extra = mcmc.get_extra_fields()
        self.divergences = int(np.asarray(extra["diverging"]).sum())
        self.mcmc = mcmc

    def zscore(self, x):
        return (np.asarray(x) - self.z_mean) / self.z_sd

    def residual_load_scenarios(self, month: str) -> list[np.ndarray]:
        """Daily residual load paths for `month` taken from the same calendar month of
        recent complete years, one path per year."""
        y, m = int(month[:4]), int(month[4:])
        n = calendar.monthrange(y, m)[1]
        h = self.history
        paths = []
        for year in sorted(h["day"].dt.year.unique(), reverse=True):
            part = h[(h["day"].dt.year == year) & (h["day"].dt.month == m)]["residual_load_gw"].to_numpy()
            if len(part) >= 28 and year < y:
                paths.append(np.resize(part, n))  # pad Feb 29 etc. by wrapping
            if len(paths) == CLIMATE_YEARS:
                break
        return paths

    def forecast_month(self, month: str, seed=1) -> np.ndarray:
        """Posterior predictive draws of the month's base index."""
        rng = np.random.default_rng(seed)
        y, m = int(month[:4]), int(month[4:])
        days = pd.date_range(dt.date(y, m, 1), periods=calendar.monthrange(y, m)[1], freq="D")
        weekend = (days.dayofweek >= 5).astype(float)
        last_week = self.n_weeks - 1
        steps = (((days - self.day0).days // 7) - last_week).to_numpy()  # weeks ahead, >= 1
        steps = np.maximum(steps, 1)
        paths = self.residual_load_scenarios(month)
        p = self.post
        S = len(p["nu"])
        H = int(steps.max())

        def carry(x_last, tau):  # random-walk paths over H weeks, shape (S, H)
            return x_last[:, None] + np.cumsum(tau[:, None] * rng.standard_normal((S, H)), axis=1)

        level = carry(p["level"][:, last_week], p["tau_level"])[:, steps - 1]
        beta = carry(p["beta"][:, last_week], p["tau_beta"])[:, steps - 1]
        sig = np.exp(carry(p["logsig"][:, last_week], p["tau_logsig"])[:, steps - 1])
        z = np.stack([self.zscore(paths[i]) for i in rng.integers(len(paths), size=S)])
        mu = level + beta * z + p["gamma"][:, None] * weekend[None, :]
        noise = rng.standard_t(p["nu"][:, None], size=mu.shape) * sig
        return (mu + noise).mean(axis=1)


# ---------------------------------------------------------------- scoring

def crps(draws: np.ndarray, x: float) -> float:
    """Continuous ranked probability score from samples (lower is better)."""
    d = np.sort(draws)
    n = len(d)
    return float(np.mean(np.abs(d - x)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * d) / n**2)


def summary(draws: np.ndarray) -> dict:
    q = np.percentile(draws, [5, 25, 50, 75, 95])
    return {"mean": float(draws.mean()), "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]}


def next_month(month: str, k: int = 1) -> str:
    y, m = int(month[:4]), int(month[4:])
    y, m = divmod(y * 12 + m - 1 + k, 12)
    return f"{y}{m + 1:02d}"


# ---------------------------------------------------------------- runs

def run_forecast(daily, index, futures):
    origin = daily["day"].max()
    fit = Fit(daily, origin, samples=1000, warmup=1000)
    print(f"Fit through {origin.date()}: {fit.n_weeks} weeks, {fit.seconds:.0f}s, {fit.divergences} divergences")
    last_trade = futures["trade_date"].max()
    curve = futures[futures["trade_date"] == last_trade].set_index("month")["settlement"]
    rows = []
    cur = origin.strftime("%Y%m")
    for k in (1, 2, 3):
        month = next_month(cur, k)
        draws = fit.forecast_month(month)
        s = summary(draws)
        fut = curve.get(month)
        s.update({"month": month, "futures": fut, "futures_trade_date": str(last_trade.date()),
                  "p_futures_above_model": float((draws < fut).mean()) if fut is not None else None})
        rows.append(s)
        np.save(OUT / f"draws_{month}.npy", draws)
    res = pd.DataFrame(rows).set_index("month")
    res.to_csv(OUT / "forecast.csv")
    p = fit.post
    params = {k: summary(p[k]) for k in ("tau_level", "tau_beta", "tau_logsig", "gamma", "nu")}
    params["level_last_week"] = summary(p["level"][:, -1])
    params["beta_last_week"] = summary(p["beta"][:, -1])
    params["sigma_last_week"] = summary(np.exp(p["logsig"][:, -1]))
    (OUT / "params.json").write_text(json.dumps({"origin": str(origin.date()), "weeks": fit.n_weeks,
                                                 "divergences": fit.divergences, "seconds": round(fit.seconds),
                                                 "params": params}, indent=2, default=float))
    pd.DataFrame({"week_start": fit.day0 + pd.to_timedelta(np.arange(fit.n_weeks) * 7, "D"),
                  "level_mean": p["level"].mean(0), "beta_mean": p["beta"].mean(0),
                  "sigma_mean": np.exp(p["logsig"]).mean(0)}).to_csv(OUT / "states.csv", index=False)
    print(res.round(1).to_string())
    return fit, res


def run_backtest(daily, index, months_back: int):
    """Refit at the last day of each month and forecast the following month."""
    index = index.set_index("month")
    complete = index[index["complete"]].index.tolist()
    targets = complete[-months_back:]
    rows = []
    for target in targets:
        prev = next_month(target, -1)
        origin = pd.Timestamp(dt.date(int(prev[:4]), int(prev[4:]), calendar.monthrange(int(prev[:4]), int(prev[4:]))[1]))
        fit = Fit(daily, origin, samples=400, warmup=400)
        draws = fit.forecast_month(target)
        actual = float(index.loc[target, "base_index"])
        naive_val = float(daily[(daily["day"] > origin - pd.Timedelta(days=30)) & (daily["day"] <= origin)]["base_price"].mean())
        s = summary(draws)
        s.update({"target": target, "origin": str(origin.date()), "actual": actual, "naive_30d": naive_val,
                  "crps": crps(draws, actual), "abs_err_mean": abs(s["mean"] - actual),
                  "abs_err_naive": abs(naive_val - actual),
                  "in_50": s["p25"] <= actual <= s["p75"], "in_90": s["p05"] <= actual <= s["p95"],
                  "pit": float((draws < actual).mean()), "divergences": fit.divergences,
                  "seconds": round(fit.seconds)})
        rows.append(s)
        print(f"{target}: actual {actual:6.1f}  model {s['mean']:6.1f} [{s['p05']:6.1f}, {s['p95']:6.1f}]  "
              f"naive {naive_val:6.1f}  pit {s['pit']:.2f}  ({fit.seconds:.0f}s, {fit.divergences} div)")
    bt = pd.DataFrame(rows).set_index("target")
    bt.to_csv(OUT / "backtest.csv")
    print(f"MAE model {bt.abs_err_mean.mean():.1f}, naive {bt.abs_err_naive.mean():.1f}; "
          f"50% coverage {bt.in_50.mean():.0%}, 90% coverage {bt.in_90.mean():.0%}; CRPS {bt.crps.mean():.1f}")
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, default=0, help="months to backtest (0 = skip)")
    ap.add_argument("--no-forecast", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build(DB).close()
    daily, index, futures = load_daily(), load_index(), load_futures()
    if not args.no_forecast:
        run_forecast(daily, index, futures)
    if args.backtest:
        run_backtest(daily, index, args.backtest)


if __name__ == "__main__":
    main()
