"""Checks for trading.py: reading Bonneville's columns, daily profiles, the kinds of day, the regime model, spells
and the day-ahead inputs. No downloads."""
import numpy as np
import pandas as pd

import trading

UTC = "UTC Time at End of Hour"


def balance_file(path, columns, rows):
    pd.DataFrame(rows, columns=["Balancing Authority", UTC] + columns).to_csv(path, index=False)
    return path


def test_bpat_rows_take_adjusted_columns_under_old_and_new_names(tmp_path):
    old = balance_file(tmp_path / "old.csv", [
        "Demand Forecast (MW)", "Demand (MW)", "Demand (MW) (Adjusted)", "Net Generation (MW) from Hydropower and "
        "Pumped Storage (Adjusted)", "Net Generation (MW) from Wind (Adjusted)"], [
        ("BPAT", "07/01/2019 8:00:00 AM", "6,000", "5,900", "5,950", "8,000", "700"),
        ("CISO", "07/01/2019 8:00:00 AM", "1", "1", "1", "1", "1")])
    new = balance_file(tmp_path / "new.csv", [
        "Demand Forecast (MW)", "Demand (MW) (Adjusted)", "Net Generation (MW) from Hydropower Excluding Pumped "
        "Storage (Adjusted)", "Net Generation (MW) from Wind with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind without Integrated Battery Storage (Adjusted)"], [
        ("BPAT", "07/01/2025 8:00:00 AM", "7,000", "6,900", "9,000", "5", "800")])
    a, b = trading.bpat_rows(old), trading.bpat_rows(new)
    assert a.iloc[0].to_dict() == {"demand_forecast": 6000, "demand": 5950, "hydro": 8000, "wind": 700}
    assert b.iloc[0].to_dict() == {"demand_forecast": 7000, "demand": 6900, "hydro": 9000, "wind": 800}
    assert list(a.index) == [pd.Timestamp("2019-07-01 08:00")]


def test_a_file_without_fuel_columns_leaves_them_missing(tmp_path):
    path = balance_file(tmp_path / "early.csv", ["Demand Forecast (MW)", "Demand (MW) (Adjusted)"],
                        [("BPAT", "07/01/2017 8:00:00 AM", "6,000", "5,950")])
    row = trading.bpat_rows(path).iloc[0]
    assert row.demand == 5950 and np.isnan(row.hydro) and np.isnan(row.wind)


def hourly(local_start, mw):
    """Hourly flows labelled, like EIA's, by the Pacific time each hour ends."""
    local = pd.DatetimeIndex(local_start) + pd.Timedelta(hours=1)
    return pd.DataFrame({"local": local, "mw": mw}, index=local + pd.Timedelta(hours=8))


def test_daily_profiles_drop_short_days_and_average_the_repeated_hour():
    hours = list(pd.date_range("2024-01-01", periods=48, freq="h"))
    hours.remove(pd.Timestamp("2024-01-02 05:00"))  # a 23-hour day
    hours.insert(3, pd.Timestamp("2024-01-01 02:00"))  # a repeated hour
    mw = np.arange(len(hours), dtype=float)
    p = trading.daily_profiles(hourly(hours, mw))
    assert list(p.index) == [pd.Timestamp("2024-01-01")] and p.shape == (1, 24)
    assert p.loc["2024-01-01", 2] == (2 + 3) / 2


def test_patterns_are_numbered_from_exporting_to_importing():
    rng = np.random.default_rng(0)
    shape = np.where((np.arange(24) >= 9) & (np.arange(24) < 16), 400, -400)
    days = np.vstack([np.full((30, 24), 1500.0), np.full((30, 24), -1500.0), np.tile(shape, (30, 1))])
    p = pd.DataFrame(days + rng.normal(0, 50, days.shape), index=pd.date_range("2024-01-01", periods=90))
    label, centres = trading.patterns(p)
    assert label.iloc[:30].eq(2).all() and label.iloc[30:60].eq(0).all() and label.iloc[60:].eq(1).all()
    assert centres[0].mean() < centres[1].mean() < centres[2].mean()


def test_spells_keep_runs_of_at_least_four_weeks():
    weeks = pd.date_range("2022-01-03", periods=12, freq="7D")
    prob = pd.Series([0.9, 0.9, 0.1, 0.8, 0.9, 0.95, 0.99, 0.2, 0.1, 0.6, 0.7, 0.8], index=weeks)
    s = trading.spells(prob)
    assert len(s) == 1 and s.start[0] == weeks[3] and s.weeks[0] == 4 and s.end[0] == weeks[6] + pd.Timedelta(days=6)


def test_regime_model_finds_a_long_spell_of_heavy_importing():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2017-05-15", "2024-12-31", freq="h")
    mw = np.where((hours >= "2022-09-01") & (hours < "2024-06-01"), 900.0, -500.0) + rng.normal(0, 300, len(hours))
    anomaly, prob, fit, params, heavy = trading.regimes(hourly(hours, mw))
    runs = trading.spells(prob)
    longest = runs.loc[runs.weeks.idxmax()]
    assert abs((longest.start - pd.Timestamp("2022-09-01")).days) <= 14
    assert abs((longest.end - pd.Timestamp("2024-06-01")).days) <= 14


def test_day_ahead_inputs_use_only_earlier_days():
    start = pd.Series(pd.date_range("2024-03-01", periods=9 * 24, freq="h"))
    d = pd.DataFrame({"start": start, "mw": start.dt.day * 100.0 + start.dt.hour,
                      "bpat_hydro": start.dt.day * 10.0, "bpat_wind": 1.0})
    x = trading.day_ahead_inputs(d)
    row = x[x.start == pd.Timestamp("2024-03-09 05:00")].iloc[0]
    assert row.same_hour_yesterday == 805 and row.same_hour_last_week == 205
    assert row.yesterday_mean == 811.5 and row.yesterday_last == 823 and row.bpat_hydro_yesterday == 80
    changed = trading.day_ahead_inputs(d.assign(mw=d.mw.where(d.start.dt.day < 9, -1e6)))
    assert changed[trading.DAY_AHEAD[:4]].equals(x[trading.DAY_AHEAD[:4]])  # the day's own flows are never used


def test_usual_is_the_training_average_for_the_month_and_hour():
    train = pd.DataFrame({"month": [1, 1, 1, 2], "hour": [5.0, 5.0, 6.0, 5.0], "mw": [100.0, 300.0, 50.0, 7.0]})
    test = pd.DataFrame({"month": [1, 1, 3], "hour": [5.0, 6.0, 5.0]})
    assert np.allclose(trading.usual(train, test), [200, 50, np.nan], equal_nan=True)


def test_errors_are_weighted_by_hours():
    errors = pd.DataFrame({"hours": [100, 300], "a": [10.0, 30.0]}, index=[2025, 2026])
    assert trading.weighted(errors)["a"] == 25
