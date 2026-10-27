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


def test_net_flow_is_named_by_its_sign():
    assert dashboard.flow(2.5) == ("net imports", 2.5)
    assert dashboard.flow(-2.7) == ("net exports", 2.7)


def test_checks_pass_clean_data_and_catch_problems():
    b, t, d = frames()
    assert dashboard.problems(b, t, d) == []
    assert dashboard.problems(b, t, d, published_through="2026-07") == []

    assert dashboard.problems(b, t.drop(pd.Period("2025-11", "M")), d) == ["missing values in the last 24 months"]

    odd = b.copy()
    odd.loc["2026-03", "hydro"] = 20_000.0
    assert dashboard.problems(odd, t, d) == ["hydro output outside 1,500 to 9,000 GWh a month"]

    pricey = t.copy()
    pricey.loc["2026-01", "import_value"] = 500.0  # $5,000 per MWh
    assert dashboard.problems(b, pricey, d) == ["trade prices outside $0 to $1,000 per MWh"]

    old_b, old_t, old_d = frames(end="2026-06")
    assert dashboard.problems(old_b, old_t, old_d, published_through="2026-07") == [
        "data end 2026-06 is older than the published 2026-07"]
