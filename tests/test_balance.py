"""Checks for the data handling and models in balance.py. No downloads: every input is built here."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import balance


def statcan_rows(column, values, geo="British Columbia", **fixed):
    """A long-format Statistics Canada table: one row per (month, member) with a VALUE."""
    return pd.DataFrame([{"GEO": geo, "REF_DATE": month, column: member, **fixed, "VALUE": v}
                         for (month, member), v in values.items()])


def test_balance_joins_old_and_new_tables_in_gwh(monkeypatch):
    old = statcan_rows("Electric power, components", {
        ("2007-12", "Total available"): 6_000_000.0, ("2007-12", "Overall total generation"): 5_000_000.0,
        ("2007-12", "Total hydro generation"): 4_000_000.0, ("2007-12", "Total receipts"): 1_500_000.0,
        ("2007-12", "Total deliveries"): 500_000.0})
    old = pd.concat([old, statcan_rows("Electric power, components", {("2007-12", "Total available"): 1.0},
                                       geo="Alberta")])
    new = statcan_rows("Electric power, components", {
        ("2008-01", "Total electricity available for use within specific geographic border"): 7_000_000.0,
        ("2008-01", "Total generation"): 8_000_000.0, ("2008-01", "Total receipts"): 200_000.0,
        ("2008-01", "Total deliveries"): 1_200_000.0})
    types = pd.concat([
        statcan_rows("Type of electricity generation", {("2008-01", "Hydraulic turbine"): 7_500_000.0},
                     **{"Class of electricity producer": "Total all classes of electricity producer"}),
        statcan_rows("Type of electricity generation", {("2008-01", "Hydraulic turbine"): 99.0},
                     **{"Class of electricity producer": "Electricity producers, industries"}),
    ])
    tables = {"25100001": old, "25100016": new, "25100015": types}
    monkeypatch.setattr(balance, "statcan", lambda pid: tables[pid])

    b = balance.balance()
    assert list(b.index.astype(str)) == ["2007-12", "2008-01"]
    assert b.demand.tolist() == [6000.0, 7000.0]  # MWh -> GWh, and only BC
    assert b.hydro.tolist() == [4000.0, 7500.0]  # only the all-producers total
    assert b.net_imports.tolist() == [1000.0, -1000.0]
    assert b.other_generation.tolist() == [1000.0, 500.0]


def test_trade_drops_total_rows_and_reads_month_first_dates(monkeypatch, tmp_path):
    rows = [
        ("07/01/2026", "Exports", "British Columbia", "Washington", 1000.0, 50_000.0),
        ("07/01/2026", "Exports", "British Columbia", "California", 3000.0, 150_000.0),
        ("07/01/2026", "Exports", "British Columbia", "Total", 4000.0, 200_000.0),
        ("07/01/2026", "Imports", "Washington", "British Columbia", 2000.0, 40_000.0),
        ("07/01/2026", "Imports", "Total", "British Columbia", 2000.0, 40_000.0),
        ("07/01/2026", "Exports", "Ontario", "New York", 9999.0, 1.0),
    ]
    path = tmp_path / "trade.csv"
    pd.DataFrame(rows, columns=["Period", "Activity", "Source", "Destination", "Energy (MW.h)",
                                "Total Value (CAN$)"]).to_csv(path, index=False)
    monkeypatch.setattr(balance, "cached", lambda url, name: path)

    t = balance.trade()
    assert list(t.index.astype(str)) == ["2026-07"]  # July, not 7 January
    assert t.export_gwh.iloc[0] == 4.0 and t.import_gwh.iloc[0] == 2.0
    assert t.export_value.iloc[0] == 0.2 and t.import_value.iloc[0] == 0.04  # $ millions


def test_degree_days_split_at_18c_scale_partial_months_and_drop_thin_ones():
    days = pd.date_range("2020-01-01", "2020-03-31")
    t = pd.Series(8.0, index=days)
    t[days[(days.month == 1) & (days.day == 1)]] = 25.0  # a warm day: 7 cooling degrees, no heating
    t = t.drop(days[(days.month == 2) & (days.day > 14)])  # February: 14 days of readings
    t = t.drop(days[(days.month == 3) & (days.day > 13)])  # March: 13 days, too few
    dd = balance.degree_days(t)
    assert np.isclose(dd.hdd[pd.Period("2020-01")], 30 * 10) and np.isclose(dd.cdd[pd.Period("2020-01")], 7)
    assert np.isclose(dd.hdd[pd.Period("2020-02")], 29 * 10)  # scaled to the full (leap) month
    assert pd.Period("2020-03") not in dd.index


def test_a_missing_month_is_filled_with_the_station_usual_gap_to_the_others():
    months = pd.period_range("2000-01", "2002-12", freq="M")
    coast = pd.DataFrame({"hdd": 300.0, "cdd": 10.0}, index=months)
    stations = {"A": coast, "B": coast + 100, "C": (coast + 200).drop(pd.Period("2001-05"))}
    filled = balance.fill_from_others(stations)
    assert np.isclose(filled["C"].loc["2001-05", "hdd"], 500.0)  # others average 350, C runs 150 above
    assert filled["A"].equals(coast)


def test_normalised_load_swaps_actual_weather_for_normal_at_the_period_effect():
    months = pd.period_range("2009-12", "2010-01", freq="M")
    d = pd.DataFrame({"load": 8000.0, "hdd": [12.0, 15.0], "cdd": 0.0, "hdd_normal": 14.0, "cdd_normal": 0.0},
                     index=months)
    params = pd.Series(0.0, index=balance.weather_columns(d).columns)
    params[["hdd_1995", "hdd_2010"]] = [100.0, 150.0]
    out = balance.normalised(d, SimpleNamespace(params=params))
    assert np.isclose(out["2009-12"], 8000 + 100 * (14 - 12))  # 2009: the 1995-2009 effect
    assert np.isclose(out["2010-01"], 8000 + 150 * (14 - 15))


def test_annual_twh_keeps_full_years_only():
    months = pd.period_range("2024-01", "2025-06", freq="M")
    twh = balance.annual_twh(pd.Series(1000.0, index=months))  # 1,000 MW all year
    assert list(twh.index) == [2024] and np.isclose(twh[2024], 1000 * 24 * 366 / 1e6)


def test_drought_parts_add_up_to_the_change_in_net_imports():
    rng = np.random.default_rng(0)
    months = pd.period_range("2013-01", "2023-12", freq="M")
    b = pd.DataFrame(rng.uniform(300, 600, (len(months), 3)), index=months, columns=["demand", "hydro", "other_generation"])
    b["generation"] = b.hydro + b.other_generation
    b["net_imports"] = b.demand - b.generation
    normal_mw = b.demand * 1000 / (24 * months.days_in_month) * rng.uniform(0.95, 1.05, len(months))
    parts = balance.drought(b, normal_mw)
    assert np.allclose(parts.drop(columns="net_imports").sum(axis=1), parts.net_imports)
    assert np.isclose(parts.net_imports.loc[2013:2022].mean(), 0)


def test_trade_by_year_prices_are_dollars_per_mwh():
    months = pd.period_range("2024-01", "2024-12", freq="M")
    t = pd.DataFrame({"export_gwh": 100.0, "import_gwh": 50.0, "export_value": 10.0, "import_value": 2.0}, index=months)
    y = balance.trade_by_year(t)
    assert np.isclose(y.export_price[2024], 100.0) and np.isclose(y.import_price[2024], 40.0)
    assert np.isclose(y.net_value[2024], 12 * 8.0)


def test_time_varying_model_recovers_a_drifting_heating_effect():
    rng = np.random.default_rng(1)
    months = pd.period_range("1980-01", "2019-12", freq="M")
    season = np.cos(2 * np.pi * (months.month - 1) / 12)
    heating = pd.Series(np.clip(9 + 7 * season + rng.normal(0, 1.5, len(months)), 0, None), index=months)
    cooling = pd.Series(np.clip(-season * 1.5 + rng.normal(0, 0.5, len(months)), 0, None), index=months)
    true_effect = pd.Series(np.repeat(np.linspace(50, 150, 40), 12), index=months)
    error = np.zeros(len(months))
    for t in range(1, len(months)):
        error[t] = 0.5 * error[t - 1] + rng.normal(0, 60)
    load = pd.Series(5000 + 5 * np.arange(len(months)) + true_effect * heating + 200 * cooling + error, index=months)
    calendar_X = pd.get_dummies(months.month, prefix="month", drop_first=True, dtype=float)

    fit = balance.TimeVaryingLoad(load, heating, cooling, calendar_X).fit(disp=False, maxiter=2000)
    estimate = pd.Series(fit.smoothed_state[2], index=months)
    decade_error = (estimate - true_effect).groupby(months.year // 10).mean().abs()
    assert decade_error.max() < 15, decade_error
