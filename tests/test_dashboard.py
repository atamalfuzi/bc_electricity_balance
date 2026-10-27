"""Checks for the dashboard's calculations and its checks before publishing. No downloads."""
import numpy as np
import pandas as pd

import dashboard


def frames(end="2026-07"):
    months = pd.period_range("2010-01", end, freq="M")
    b = pd.DataFrame({"demand": 5000.0, "generation": 5000.0, "hydro": 4000.0, "receipts": 500.0,
                      "deliveries": 500.0, "other_generation": 1000.0, "net_imports": 0.0}, index=months)
    t = pd.DataFrame({"export_gwh": 100.0, "import_gwh": 100.0, "export_value": 8.0, "import_value": 5.0}, index=months)
    d = pd.DataFrame({"hdd": 10.0, "cdd": 0.5}, index=months)
    return b, t, d


def hourly_flows(end="2026-10-26 00:00"):
    """Hourly flows labelled, like EIA's, by the Pacific time each hour ends."""
    local = pd.date_range("2018-01-01 01:00", end, freq="h")
    return pd.DataFrame({"local": local, "mw": 0.0}, index=local + pd.Timedelta(hours=8))


def test_summary_compares_hydro_with_the_baseline_and_sums_the_last_twelve_months():
    b, t, d = frames()
    b.loc["2025-08":"2026-07", "hydro"] = 4400.0  # the last 12 months, 10% above the 2013-2022 average
    b.loc["2025-08":"2026-07", "net_imports"] = -250.0
    b.loc["2024-08":"2025-07", "net_imports"] = 500.0
    mw = pd.Series(5000.0, index=b.index) * 1000 / (24 * b.index.days_in_month)  # 5,000 GWh every month
    s = dashboard.summary(b, t, mw * 1.25, mw)
    assert s["month"] == "2026-07" and np.isclose(s["demand_vs_expected"], 1 / 1.25 - 1)
    assert np.isclose(s["hydro_last_12_vs_normal"], 0.1)
    assert s["net_imports_last_12_twh"] == -3.0 and s["net_imports_previous_12_twh"] == 6.0
    assert np.isclose(s["net_imports_share_of_demand"], -0.05)
    assert s["us_trade_net_last_12_million"] == 36 and s["export_price_last_12"] == 80.0


def test_hourly_summary_covers_the_last_full_month_against_the_same_month_before():
    f = hourly_flows()  # data end partway through October, so September is the last full month
    start = f.local - pd.Timedelta(hours=1)
    september = (start.dt.month == 9)
    f.loc[september & (start.dt.year == 2026), "mw"] = -1000.0
    f.loc[september & (start.dt.year == 2026) & start.dt.hour.between(17, 20), "mw"] = -2000.0
    f.loc[september & start.dt.year.between(2018, 2022) & (start.dt.hour < 6), "mw"] = 300.0  # importing overnight
    h = dashboard.hourly(f)
    assert h["hourly_month"] == "2026-09"
    assert h["hourly_share_importing"] == 0 and h["hourly_share_importing_baseline"] == 0.25
    assert h["hourly_evening_mw"] == -2000 and h["hourly_overnight_mw"] == -1000
    assert h["hourly_net_gwh"] == -(20 * 1000 + 4 * 2000) * 30 / 1000  # 20 hours a day at 1,000 MW out, 4 at 2,000


def test_last_full_month_counts_a_month_that_ends_with_the_data():
    assert dashboard.last_full_month(hourly_flows(end="2026-10-01 00:00")) == pd.Period("2026-09", "M")
    assert dashboard.last_full_month(hourly_flows(end="2026-09-30 23:00")) == pd.Period("2026-08", "M")


def test_net_flow_is_named_by_its_sign():
    assert dashboard.flow(2.5) == ("net imports", 2.5)
    assert dashboard.flow(-2.7) == ("net exports", 2.7)
    assert dashboard.mw(-1148) == "1,148 MW out" and dashboard.mw(650) == "650 MW in"


def test_checks_pass_clean_data_and_catch_problems():
    b, t, d = frames()
    f = hourly_flows()
    assert dashboard.problems(b, t, d, f) == []
    assert dashboard.problems(b, t, d, f, published_through="2026-07") == []

    assert dashboard.problems(b, t.drop(pd.Period("2025-11", "M")), d, f) == ["missing values in the last 24 months"]

    odd = b.copy()
    odd.loc["2026-03", "hydro"] = 20_000.0
    assert dashboard.problems(odd, t, d, f) == ["hydro output outside 1,500 to 9,000 GWh a month"]

    pricey = t.copy()
    pricey.loc["2026-01", "import_value"] = 500.0  # $5,000 per MWh
    assert dashboard.problems(b, pricey, d, f) == ["trade prices outside $0 to $1,000 per MWh"]

    gap = f[(f.local < "2026-09-10") | (f.local > "2026-09-13")]  # three days of September missing
    assert dashboard.problems(b, t, d, gap) == ["hourly flows missing for more than 5% of 2026-09"]
    spike = f.copy()
    spike.loc[spike.local == "2026-09-15 12:00", "mw"] = 5000.0
    assert dashboard.problems(b, t, d, spike) == ["hourly flows beyond 4,000 MW either way"]

    old_b, old_t, old_d = frames(end="2026-06")
    assert dashboard.problems(old_b, old_t, old_d, f, published_through="2026-07") == [
        "data end 2026-06 is older than the published 2026-07"]
