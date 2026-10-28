"""Checks for rivers.py: paging through a gauge's record, annual means, return periods and the inflow model.
No downloads."""
import numpy as np
import pandas as pd
from scipy import stats

import rivers


def test_river_pages_through_the_record(monkeypatch, tmp_path):
    days = pd.date_range("2020-01-01", periods=7)
    pages = {}
    for offset in (0, 3, 6):  # pages of three rows: full, full, then one row
        path = tmp_path / f"page_{offset}.csv"
        pd.DataFrame({"DATE": days[offset:offset + 3].strftime("%Y-%m-%d"),
                      "DISCHARGE": [float(i) if i != 4 else None for i in range(offset, min(offset + 3, 7))]}).to_csv(path, index=False)
        pages[f"river_X_{offset}.csv"] = path
    monkeypatch.setattr(rivers, "PAGE", 3)
    monkeypatch.setattr(rivers.balance, "cached", lambda url, name: pages[name])
    flow = rivers.river("X")
    assert list(flow.index) == list(days.delete(4)) and flow.tolist() == [0, 1, 2, 3, 5, 6]  # a missing day is dropped


def test_annual_mean_needs_almost_a_full_year():
    days = pd.date_range("2022-01-01", "2023-12-31")
    flow = pd.Series(100.0, index=days).drop(days[(days.year == 2023) & (days.month == 1)])  # 2023: 334 days
    assert rivers.annual_mean(flow).to_dict() == {2022: 100.0}


def test_return_period_reads_the_lognormal_and_brackets_it():
    sample = stats.lognorm.rvs(0.15, scale=2700, size=3000, random_state=0)
    one_in_100 = stats.lognorm.ppf(0.01, 0.15, scale=2700)
    years, low, high = rivers.return_period(sample, one_in_100, draws=200)
    assert 80 < years < 125 and low < years < high
    assert 1.8 < rivers.return_period(sample, 2700, draws=50)[0] < 2.2  # the median: every other year


def test_inflow_model_recovers_its_coefficients_and_holds_each_year_out():
    rng = np.random.default_rng(0)
    years = pd.RangeIndex(1979, 2025)
    inflow = pd.Series(rng.normal(1, 0.1, len(years)), index=years)
    demand = pd.Series(np.linspace(40, 70, len(years)), index=years)
    hydro = -20 + 30 * inflow + 10 * inflow.shift(1) + 0.6 * demand + rng.normal(0, 0.5, len(years))
    fit, d = rivers.inflow_model(pd.DataFrame({"hydro": hydro, "demand": demand}), inflow)
    assert d.index.min() == 1980  # 1979 has no year before it
    assert np.allclose(fit.params[["inflow", "inflow_last_year", "demand"]], [30, 10, 0.6], atol=[3, 3, 0.05])
    in_sample, held_out = (d.hydro - d.predicted).abs().mean(), (d.hydro - d.held_out).abs().mean()
    assert held_out > in_sample  # each year predicted without itself does worse


def test_low_water_splits_out_the_part_normal_inflows_would_restore():
    years = pd.RangeIndex(2013, 2025)
    annual = pd.DataFrame({"hydro": 60.0, "net_imports": -3.0}, index=years)
    annual.loc[2023, ["hydro", "net_imports"]] = [49.0, 11.0]
    d = pd.DataFrame({"hydro": annual.hydro, "inflow": 1.0, "inflow_last_year": 1.0, "predicted": 60.0}, index=years)
    d.loc[2023, ["inflow", "predicted"]] = [0.8, 54.0]

    class Fit:  # 30 TWh per unit of this year's inflow
        @staticmethod
        def t_test(weights):
            effect = float(weights @ np.array([0, 30, 10, 0]))
            return type("T", (), {"effect": [effect], "conf_int": lambda self, alpha: [[effect - 1, effect + 1]]})()

    parts = rivers.low_water(Fit, d, annual, years=(2023,), baseline=(2013, 2022))
    assert np.isclose(parts.loc[2023, "hydro_change"], -11) and np.isclose(parts.loc[2023, "from_low_inflows"], 6)
    assert np.isclose(parts.loc[2023, "net_imports_change"], 14) and np.isclose(parts.loc[2023, "unexplained"], -5)
