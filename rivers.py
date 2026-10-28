"""From river water to power: how rare the 2023-2024 low flows were over 112 years, and how much of the fall in BC's
hydro output the low inflows to its two biggest reservoirs explain.

Data: the Water Survey of Canada's daily mean river flows (HYDAT) and Statistics Canada's monthly electricity tables.

Run from the repository folder, after balance.py: python rivers.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import balance
from balance import BLUE, GREY, INK, MUTED, ORANGE, RED

FRASER = "08MF005"  # Fraser River at Hope: the longest record, from 1912
INFLOWS = {"Columbia at Donald (into Kinbasket Lake, Mica Dam)": "08NB005",
           "Finlay above Akie (into Williston Lake, W.A.C. Bennett Dam)": "07EA005"}
NORMAL = (1991, 2020)
PAGE = 10000  # the most rows the service returns per request
SOURCE = "Water Survey of Canada daily mean discharge (HYDAT), via the Government of Canada's hydrometric API."


def river(station):
    """Daily mean flow (cubic metres a second) at a Water Survey of Canada gauge, over its full record."""
    parts, offset = [], 0
    while True:
        url = ("https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?f=csv&sortby=DATE"
               f"&limit={PAGE}&offset={offset}&STATION_NUMBER={station}&properties=DATE,DISCHARGE")
        part = pd.read_csv(balance.cached(url, f"river_{station}_{offset}.csv"))
        parts.append(part)
        if len(part) < PAGE:
            break
        offset += PAGE
    t = pd.concat(parts)
    return t.set_index(pd.to_datetime(t.DATE)).DISCHARGE.dropna().sort_index()


def annual_mean(daily, min_days=355):
    """Calendar-year mean flow, for years with at least min_days readings."""
    g = daily.groupby(daily.index.year)
    return g.mean()[g.count() >= min_days]


def return_period(flows, x, draws=2000, seed=0):
    """Years between annual flows this low or lower: from a lognormal fitted to the record, with a 90% band from
    refitting to bootstrap resamples of the years."""
    def years(sample):
        shape, loc, scale = stats.lognorm.fit(sample, floc=0)
        return 1 / stats.lognorm.cdf(x, shape, loc, scale)
    rng = np.random.default_rng(seed)
    boot = [years(rng.choice(flows, len(flows))) for _ in range(draws)]
    return years(flows), *np.percentile(boot, [5, 95])


def inflow_model(annual, inflow):
    """Annual hydro output (TWh) on this year's and last year's inflow index and demand, Newey-West errors; with
    each year's prediction from a fit that leaves that year out."""
    d = pd.DataFrame({"hydro": annual.hydro, "demand": annual.demand, "inflow": inflow,
                      "inflow_last_year": inflow.shift(1)}).loc[1980:].dropna()
    X = sm.add_constant(d[["inflow", "inflow_last_year", "demand"]])
    fit = sm.OLS(d.hydro, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    held_out = pd.Series({y: sm.OLS(d.hydro.drop(y), X.drop(y)).fit().predict(X.loc[[y]]).iloc[0] for y in d.index})
    return fit, d.assign(predicted=fit.fittedvalues, held_out=held_out)


def low_water(fit, d, annual, years=(2023, 2024), baseline=balance.BASELINE):
    """For each drought year, against the baseline average: the fall in hydro output, the part the low inflows
    explain (output with normal inflows, all else as it was, with a 90% range) and the rise in net imports."""
    base = annual.loc[baseline[0]:baseline[1]].mean()
    rows = {}
    for y in years:
        effect = fit.t_test(np.array([0, 1 - d.inflow[y], 1 - d.inflow_last_year[y], 0]))
        lo, hi = effect.conf_int(alpha=0.10)[0]
        rows[y] = {"hydro_change": annual.hydro[y] - base.hydro, "from_low_inflows": float(effect.effect[0]),
                   "low_90": lo, "high_90": hi, "net_imports_change": annual.net_imports[y] - base.net_imports,
                   "unexplained": d.hydro[y] - d.predicted[y]}
    return pd.DataFrame(rows).T.rename_axis("year")


def main():
    annual = balance.full_years(balance.balance()) / 1000
    fraser = annual_mean(river(FRASER))
    gauges = pd.DataFrame({name: annual_mean(river(s)) for name, s in INFLOWS.items()})
    inflow = (gauges / gauges.loc[NORMAL[0]:NORMAL[1]].mean()).mean(axis=1, skipna=False)  # 1 = normal

    rank, two_year = fraser.rank(), fraser.rolling(2).mean()
    tau, p_trend = stats.kendalltau(fraser.index, fraser.values)
    slope, intercept, *_ = stats.theilslopes(fraser.values, fraser.index)
    drought_years = [y for y in (2023, 2024) if y in fraser.index]
    periods = {y: return_period(fraser.values, fraser[y]) for y in drought_years}
    fit, d = inflow_model(annual, inflow)
    rmse = np.sqrt(((d.hydro - d.held_out) ** 2).mean())
    parts = low_water(fit, d, annual)

    pd.concat({"fraser_at_hope": fraser, "columbia_at_donald": gauges.iloc[:, 0], "finlay_above_akie": gauges.iloc[:, 1],
               "inflow_index": inflow}, axis=1).rename_axis("year").round(3).to_csv(balance.OUTPUT / "rivers.csv")
    d.rename_axis("year").round(2).to_csv(balance.OUTPUT / "water_to_power.csv")
    parts.round(2).to_csv(balance.OUTPUT / "low_water.csv")
    print(f"Fraser at Hope {fraser.index.min()}-{fraser.index.max()} ({len(fraser)} years), median {fraser.median():.0f} m3/s")
    for y in drought_years:
        t, lo, hi = periods[y]
        print(f"  {y}: {fraser[y]:.0f} m3/s, {fraser[y] / fraser.median() - 1:+.0%}, rank {rank[y]:.0f} from lowest; "
              f"once in {t:.0f} years (90%: {lo:.0f}-{hi:.0f})")
    print(f"  Lowest two-year runs (ending): {two_year.nsmallest(3).round(0).to_dict()}")
    print(f"  Trend: Theil-Sen {slope * 100:+.0f} m3/s a century, Kendall tau {tau:.2f}, p = {p_trend:.2f}")
    print("Inflow index (1 = 1991-2020 normal):", inflow.loc[2018:].round(2).to_dict())
    print(fit.summary().tables[1])
    print(f"R2 {fit.rsquared:.2f}; held-out error {rmse:.1f} TWh over {len(d)} years")
    print(parts.round(1).to_string())

    plt.rcParams.update(balance.STYLE)
    # How rare: 112 years of the Fraser, and return periods
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.17, wspace=0.2)
    low = fraser.nsmallest(5).index
    left.plot(fraser.index, fraser, color=GREY, lw=1.2, marker="o", ms=3, label="Each year")
    left.axhline(fraser.median(), color=MUTED, lw=0.8, label="Median")
    left.plot(fraser.index, intercept + slope * fraser.index, color=INK, lw=1.2, ls="--",
              label=f"Trend (Theil-Sen): none to speak of, p = {p_trend:.2f}")
    for y in low:
        colour = RED if y in drought_years else ORANGE
        left.plot(y, fraser[y], "o", color=colour, ms=7, zorder=3)
        left.annotate(str(y), (y, fraser[y]), (10, -3) if y == low.max() else (0, -15), textcoords="offset points",
                      ha="left" if y == low.max() else "center", fontsize=9, color=colour)
    left.set_ylim(1700, fraser.max() * 1.1)
    left.set_ylabel("Mean flow over the year, m³/s")
    left.legend(loc="upper left", fontsize=9, ncol=3)
    left.set_title(f"Fraser River at Hope, {fraser.index.min()}-{fraser.index.max()} (five lowest coloured)")
    n = len(fraser)
    shape, loc, scale = stats.lognorm.fit(fraser.values, floc=0)
    grid = np.logspace(0.05, np.log10(300), 100)
    right.plot(grid, stats.lognorm.ppf(1 / grid, shape, loc, scale), color=BLUE, lw=2, label="Lognormal fit")
    right.scatter((n + 1) / np.arange(1, n + 1), fraser.sort_values().values, s=14, color=GREY, zorder=3,
                  label="Each year, ranked")  # Weibull plotting positions for low flows
    for y in drought_years:
        t, lo, hi = periods[y]
        right.errorbar(t, fraser[y], xerr=[[t - lo], [hi - t]], fmt="o", color=RED, ms=6, capsize=3, zorder=4)
        right.annotate(f"{y}: 1 in {round(t, -1 if t > 50 else 0):.0f} years", (t, fraser[y]), (-12, -20),
                       textcoords="offset points", ha="right", fontsize=9, color=RED)
    right.set_xscale("log")
    right.set_xticks([1, 2, 5, 10, 20, 50, 100, 200], ["1", "2", "5", "10", "20", "50", "100", "200"])
    right.set_xlabel("Return period, years (a year this low or lower)")
    right.legend(loc="upper right", fontsize=9)
    right.set_ylim(1650, None)
    right.set_title("How rare a low year is")
    driest = two_year.idxmin() == drought_years[-1] if drought_years else False
    balance.save(fig, "river_history.png",
                 f"2023 was the Fraser's {'second-' if rank.get(2023) == 2 else ''}lowest year since {fraser.index.min()}"
                 + (", and 2023-2024 its driest two years in a row" if driest else ""),
                 "Mean flow of the Fraser River at Hope each calendar year, and how often a year that low comes around",
                 "Return periods from a lognormal fitted to all years; bars are 90% bootstrap ranges (2,000 resamples of "
                 "the years).\n" + SOURCE)

    # Inflows to the big reservoirs and BC's hydro output
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.25, wspace=0.25)
    left.plot(d.index, d.hydro, color=BLUE, lw=2.4, label="Actual hydro output")
    left.plot(d.index, d.held_out, color=GREY, lw=1.8, ls="--", label="Model, each year predicted without it")
    left.set_ylabel("TWh a year")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    left.set_title(f"Hydro output and the inflow model (R² {fit.rsquared:.2f}, held-out error {rmse:.1f} TWh)")
    x, w = np.arange(len(parts)), 0.26
    right.bar(x - w, -parts.hydro_change, w, color=BLUE, label="Fall in hydro output")
    right.bar(x, parts.from_low_inflows, w, color=GREY, label="Explained by low inflows (90% range)",
              yerr=[parts.from_low_inflows - parts.low_90, parts.high_90 - parts.from_low_inflows], capsize=3, ecolor=MUTED)
    right.bar(x + w, parts.net_imports_change, w, color=RED, label="Rise in net imports")
    right.set_xticks(x, parts.index)
    right.set_ylabel(f"TWh against {balance.BASELINE[0]}-{balance.BASELINE[1]} average")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    right.set_title("The drought years")
    balance.save(fig, "water_to_power.png",
                 f"Low river inflows explain about {parts.from_low_inflows[2023]:.0f} TWh of the "
                 f"{-parts.hydro_change[2023]:.0f} TWh fall in BC's hydro output in 2023",
                 f"Annual hydro output modelled on inflows to Kinbasket and Williston lakes (this year and last) and "
                 f"demand, {d.index.min()}-{d.index.max()}",
                 "Inflow index: mean flow of the Columbia at Donald and the Finlay above Akie, each as a share of its "
                 "1991-2020 average. Newey-West standard errors.\n" + SOURCE + " Statistics Canada tables 25-10-0001-01, "
                 "25-10-0015-01 and 25-10-0016-01.")


if __name__ == "__main__":
    main()
