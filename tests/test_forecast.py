"""Checks for forecast.py: the three forecasting methods, conformal ranges and the weather adjustment's ranges.
No downloads."""
import numpy as np
import pandas as pd

import balance
import forecast


def monthly_load(start="2000-01", end="2020-12", seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    hdd = pd.Series(np.clip(8 + 7 * np.cos(2 * np.pi * (months.month - 1) / 12), 0, None), index=months)
    cdd = pd.Series(np.clip(-np.cos(2 * np.pi * (months.month - 1) / 12), 0, None), index=months)
    load = 6000 + 2 * np.arange(len(months)) + 150 * hdd + 300 * cdd + rng.normal(0, 60, len(months))
    return pd.DataFrame({"load": load, "hdd": hdd, "cdd": cdd, "hdd_normal": 8 + 7 * np.cos(2 * np.pi * (months.month - 1) / 12),
                         "cdd_normal": cdd}, index=months)


def test_models_forecast_six_months_with_ordered_ranges():
    d = monthly_load()
    y = d.load
    ahead = pd.period_range("2021-01", periods=6, freq="M")
    out = forecast.models(y, d[["hdd", "cdd"]], forecast.normal_weather(d, ahead))
    for name in (forecast.BEST, "Exponential smoothing"):
        f = out[name]
        assert list(f.index) == list(ahead)
        assert (f.low_95 < f.low_80).all() and (f.low_80 < f["mean"]).all() and (f["mean"] < f.high_80).all() \
            and (f.high_80 < f.high_95).all()
        assert (f["mean"] / y.iloc[-12:-6].to_numpy() - 1).abs().max() < 0.1  # close to a year before
    assert np.allclose(out["Same month last year"]["mean"], y.loc["2020-01":"2020-06"])


def test_conformal_ranges_cover_as_stated_on_steady_errors():
    rng = np.random.default_rng(0)
    rows = []
    origins = pd.period_range("2010-01", periods=200, freq="M")
    for o in origins:
        for h in (1, 2):
            err = rng.normal(0, 0.03)
            rows.append({"origin": o, "month": o + h, "horizon": h, "model": "m", "mean": 100.0, "actual": 100 * (1 + err)})
    cover, widths = forecast.conformal(pd.DataFrame(rows), "m")
    assert abs(cover[80] - 0.8) < 0.06 and abs(cover[95] - 0.95) < 0.04
    assert np.allclose(widths.loc[1, 80], 0.03 * 1.2816, rtol=0.2)  # |error| below its 80th percentile


def test_weather_ranges_center_on_the_weather_effect_in_the_normalised_demand():
    d = monthly_load("1990-01", "2014-12", seed=1)  # spans the demand model's three periods
    rng = np.random.default_rng(2)  # weather that differs from normal
    d["hdd"] = (d.hdd + rng.normal(0, 1.5, len(d))).clip(0)
    d["cdd"] = (d.cdd + rng.normal(0, 0.3, len(d))).clip(0)
    fit = balance.demand_model(d)
    ranges = forecast.weather_ranges(d, fit)
    normal = balance.normalised(d, fit)
    effect = balance.annual_twh(d.load) - balance.annual_twh(normal)
    assert np.allclose(ranges.weather_twh, effect.loc[ranges.index])
    assert (ranges.low_90 < ranges.weather_twh).all() and (ranges.weather_twh < ranges.high_90).all()
