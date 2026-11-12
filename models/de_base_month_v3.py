"""Model v3: version 2 with fixes for the summer over-forecast.

Version 2 over-forecast June to October by about 12 EUR/MWh. Two fixes, each a variant of
models/de_base_month_v2.py; the current model ("version 3") is `both`, which did best in
the 24-month backtest:

  solar     residual-load scenarios use past years' weather with today's solar fleet:
            solar output from each past year is scaled by installed solar capacity now
            versus then. Plain past years overstate residual load as solar keeps growing,
            most in summer.
  seasonal  a shift in the price/cost multiple for each calendar month (12 shifts,
            centred to average zero):
            y_d ~ StudentT(nu, c_d * (m_w + season[month_d] + s_w * z_d) + gamma * weekend_d, ...)
  both      both of the above (the current model)

Usage:
  python -m models.de_base_month_v3                  # current model (both): fit, forecast the next 3 months
  python -m models.de_base_month_v3 --backtest 24
  python -m models.de_base_month_v3 --variant solar --backtest 24
Outputs go to reports/de_base_month_v3_<variant>/.
"""

import pathlib
import sys

from models import de_base_month_v2 as v2

REPORTS = pathlib.Path(__file__).resolve().parents[1] / "reports"
VARIANTS = {"solar": dict(capacity_adjusted=True), "seasonal": dict(seasonal=True),
            "both": dict(capacity_adjusted=True, seasonal=True)}


def out_dir(variant: str) -> pathlib.Path:
    return REPORTS / f"de_base_month_v3_{variant}"


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = "both"
    if "--variant" in args:
        i = args.index("--variant")
        variant = args[i + 1]
        del args[i:i + 2]
    sys.argv = [sys.argv[0]] + args
    v2.main(out=out_dir(variant), **VARIANTS[variant])
