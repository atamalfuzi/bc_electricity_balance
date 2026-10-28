"""BC's hourly trade with the United States: the kinds of trading day (k-means), spells of heavy importing (a
Markov-switching model), a day-ahead forecast of the hourly flow (linear and gradient-boosting models tested on later
years), and what drives the flow (permutation importance and partial dependence). Also writes the data behind the
write-up's interactive charts.

Data: the US Energy Information Administration's Form EIA-930 (the BC Hydro-Bonneville flow, and Bonneville's demand,
demand forecast and hydro and wind output), Statistics Canada's monthly hydro output and Vancouver's daily temperature.

Run from the repository folder, after balance.py and rivers.py: python trading.py
"""
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import balance
import rivers
from balance import BLUE, GREY, INK, MUTED, ORANGE, RED

# Data: Bonneville's demand and output ----
# EIA-930's balance files, one per half-year, about 45 MB each; past half-years are stored in flows/, as for the
# flows themselves. Generation by fuel starts in July 2018, and EIA renamed the fuel columns in mid-2024.
EIA_BALANCE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{}_{}.csv"
BPAT = Path("flows") / "bpat_hourly.csv.gz"
COLUMNS = {"demand_forecast": "Demand Forecast (MW)", "demand": "Demand (MW)", "hydro": "Hydro", "wind": "Wind"}


def pick(header, key, name):
    """A file's column for an item: EIA's adjusted series where there is one; fuels are matched by name."""
    if key in ("hydro", "wind"):
        found = [h for h in header if h.startswith("Net Generation (MW) from") and name in h
                 and "with Integrated Battery" not in h]
    else:
        found = [h for h in header if h in (name, name.replace("(MW)", "(MW) (Adjusted)"))]
    return ([h for h in found if "Adjusted" in h] or found or [None])[0]


def bpat_rows(path):
    """The Bonneville Power Administration's hourly rows from one EIA-930 balance file, indexed by UTC hour ending."""
    header = pd.read_csv(path, nrows=0).columns
    cols = {k: pick(header, k, v) for k, v in COLUMNS.items()}
    t = pd.read_csv(path, usecols=["Balancing Authority", "UTC Time at End of Hour"] + [c for c in cols.values() if c],
                    thousands=",", low_memory=False)
    t = t[t["Balancing Authority"] == "BPAT"]
    index = pd.to_datetime(t["UTC Time at End of Hour"], format=balance.TIME).rename("utc")
    return pd.DataFrame({k: pd.to_numeric(t[c], errors="coerce").to_numpy() if c else np.nan for k, c in cols.items()},
                        index=index)


def bpat(today=None):
    """Bonneville's hourly demand, demand forecast and hydro and wind output: the stored half-years and every later one."""
    past = pd.read_csv(BPAT, index_col="utc", parse_dates=["utc"])
    new = [bpat_rows(balance.cached(EIA_BALANCE.format(year, half), f"eia_balance_{year}_{half}.csv"))
           for year, half in balance.half_years(past.index.max(), today)]
    b = pd.concat([past, *new])
    return b[~b.index.duplicated(keep="last")].sort_index()


# Kinds of trading day and spells of heavy importing ----
K = 3
NAMES = ["Exporting all day", "Two-way: imports by day, exports in the evening", "Importing all day"]
COLOURS = [BLUE, GREY, RED]


def daily_profiles(f):
    """One row per day, one column per Pacific hour beginning, MW into BC. Days without all 24 hours (the spring
    clock change) are dropped, and the autumn's repeated hour is averaged."""
    start = balance.hour_start(f)
    p = pd.DataFrame({"date": start.dt.normalize(), "hour": start.dt.hour, "mw": f.mw.to_numpy()})
    return p.pivot_table(index="date", columns="hour", values="mw", aggfunc="mean").dropna()


def patterns(p):
    """k-means groups of daily profiles, numbered from most exporting to most importing, and their centres."""
    km = KMeans(K, n_init=20, random_state=0).fit(p.values)
    order = np.argsort(km.cluster_centers_.mean(axis=1))
    rank = {old: new for new, old in enumerate(order)}
    return pd.Series([rank[x] for x in km.labels_], index=p.index), km.cluster_centers_[order]


def regimes(f):
    """Weekly mean flow less the baseline years' average for its month, and the smoothed probability that each
    week is in the heavier-importing of two regimes (Markov switching in mean and variance)."""
    start = balance.hour_start(f)
    weekly = f.mw.groupby(start.dt.to_period("W")).mean()
    weekly = weekly[weekly.index.start_time >= pd.Timestamp(balance.FLOWS_START) + pd.Timedelta(days=6)]  # full weeks
    base = weekly[(weekly.index.year >= balance.FLOWS_BASELINE[0]) & (weekly.index.year <= balance.FLOWS_BASELINE[1])]
    usual = base.groupby(base.index.start_time.month).mean()
    anomaly = (weekly - usual.reindex(weekly.index.start_time.month).to_numpy()).set_axis(weekly.index.start_time)
    # The likelihood is flat near its peak, so the random restarts can move the means by a few MW between runs
    fit = MarkovRegression(anomaly.to_numpy(), k_regimes=2, trend="c", switching_variance=True).fit(search_reps=30)
    params = pd.Series(fit.params, index=fit.model.param_names)
    heavy = int(params.filter(like="const").to_numpy().argmax())
    return anomaly, pd.Series(fit.smoothed_marginal_probabilities[:, heavy], index=anomaly.index), fit, params, heavy


def spells(prob, min_weeks=4):
    """Runs of at least min_weeks weeks in which the heavy-import regime is the likelier."""
    on = prob > 0.5
    s = pd.DataFrame({"on": on, "run": (on != on.shift()).cumsum()}).groupby("run").agg(
        on=("on", "first"), start=("on", lambda x: x.index[0]), weeks=("on", "size"))
    s["end"] = s.start + pd.to_timedelta(s.weeks * 7 - 1, unit="D")
    return s[s.on & (s.weeks >= min_weeks)][["start", "end", "weeks"]].reset_index(drop=True)


# Hourly models ----
CALENDAR = ["hour", "weekday", "season_sin", "season_cos"]
DRIVERS = {"bc_water": "BC hydro output, past 12 months", "bpat_hydro": "Bonneville hydro output",
           "bpat_wind": "Bonneville wind output", "bpat_demand": "Bonneville demand",
           "temperature": "Vancouver temperature", "hour": "Hour of the day", "weekday": "Day of the week",
           "season_sin": "Time of year (sine)", "season_cos": "Time of year (cosine)"}
DAY_AHEAD = ["same_hour_yesterday", "same_hour_last_week", "yesterday_mean", "yesterday_last", "bpat_demand_forecast",
             "bpat_hydro_yesterday", "bpat_wind_yesterday", "temperature", "bc_water"] + CALENDAR


def day_ahead_inputs(d):
    """What is known at the midnight before each hour: the flow at the same hour yesterday and a week before,
    yesterday's average and last hour, and yesterday's average Bonneville hydro and wind output."""
    flow = d.drop_duplicates("start").set_index("start").mw.asfreq("h")
    day = flow.groupby(flow.index.normalize())
    hour, today = d.start.dt.floor("h"), d.start.dt.normalize()
    output = d.groupby(today)[["bpat_hydro", "bpat_wind"]].mean().shift(1)
    return d.assign(same_hour_yesterday=flow.reindex(hour - pd.Timedelta(days=1)).to_numpy(),
                    same_hour_last_week=flow.reindex(hour - pd.Timedelta(days=7)).to_numpy(),
                    yesterday_mean=day.mean().shift(1).reindex(today).to_numpy(),
                    yesterday_last=day.last().shift(1).reindex(today).to_numpy(),
                    bpat_hydro_yesterday=output.bpat_hydro.reindex(today).to_numpy(),
                    bpat_wind_yesterday=output.bpat_wind.reindex(today).to_numpy())


def frame(f, bp, b, temperature):
    """One row per hour: the flow into BC, Bonneville's demand, demand forecast and output, BC's water (hydro output
    over the 12 months to the month before, % of the 2013-2022 average, carried past the latest published month),
    Vancouver's daily temperature and the calendar, plus the day-ahead inputs."""
    d = pd.DataFrame({"mw": f.mw, "start": balance.hour_start(f)}).join(
        bp[["demand", "demand_forecast", "hydro", "wind"]].add_prefix("bpat_"), how="inner")
    year = b.hydro.rolling(12).sum()
    water = (year / year.loc["2013-12":"2022-12"].mean()).shift(1) * 100
    month = d.start.dt.to_period("M")
    water = water.reindex(pd.period_range(water.index.min(), month.max(), freq="M")).ffill()
    angle = 2 * np.pi * d.start.dt.dayofyear / 365.25
    d = d.assign(bc_water=water.reindex(month).to_numpy(),
                 temperature=temperature.reindex(d.start.dt.normalize()).to_numpy(),
                 hour=d.start.dt.hour.astype(float), weekday=d.start.dt.weekday.astype(float), month=d.start.dt.month,
                 season_sin=np.sin(angle), season_cos=np.cos(angle), year=d.start.dt.year)
    return day_ahead_inputs(d).dropna(subset=["mw", "bpat_hydro", "bpat_wind", "bpat_demand", "bc_water", "temperature"])


def linear(numbers):
    return make_pipeline(ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore"), ["hour", "weekday", "month"]),
        ("numbers", StandardScaler(), numbers)]), Ridge(alpha=1.0))


def boosted():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=0)


def usual(train, test):
    """The training years' average flow for each test hour's month and hour."""
    m = train.groupby(["month", "hour"]).mw.mean()
    return m.reindex(pd.MultiIndex.from_arrays([test.month, test.hour])).to_numpy()


def weighted(errors):
    """Mean absolute errors over all years, each year weighted by its hours."""
    return errors.drop(columns="hours").mul(errors.hours / errors.hours.sum(), axis=0).sum()


def forecast(d, first=2021):
    """Day-ahead forecasts of each year from `first`, by models trained on all earlier years: mean absolute error by
    year, and the forecasts."""
    d = d.dropna(subset=DAY_AHEAD)
    numbers = [x for x in DAY_AHEAD if x not in CALENDAR]
    rows, held = [], []
    for y in range(first, d.year.max() + 1):
        train, test = d[d.year < y], d[d.year == y]
        pred = {"Same hour yesterday": test.same_hour_yesterday.to_numpy(),
                "Usual for the month and hour": usual(train, test),
                "Linear (ridge)": linear(numbers).fit(train[DAY_AHEAD + ["month"]], train.mw)
                .predict(test[DAY_AHEAD + ["month"]]),
                "Gradient boosting": boosted().fit(train[DAY_AHEAD], train.mw).predict(test[DAY_AHEAD])}
        rows.append({"year": y, "hours": len(test), **{k: np.mean(np.abs(test.mw - v)) for k, v in pred.items()}})
        held.append(pd.DataFrame(pred, index=test.index).assign(actual=test.mw.to_numpy(), start=test.start.to_numpy()))
    return pd.DataFrame(rows).set_index("year"), pd.concat(held)


def drivers(d):
    """Gradient boosting on same-hour conditions only: its error with each year held out against the usual for the
    month and hour, then a fit to all hours read with permutation importance and partial dependence."""
    X = list(DRIVERS)
    errors = []
    for y in sorted(d.year.unique()):
        train, test = d[d.year != y], d[d.year == y]
        errors.append({"year": y, "hours": len(test), "usual": np.mean(np.abs(test.mw - usual(train, test))),
                       "boosting": np.mean(np.abs(test.mw - boosted().fit(train[X], train.mw).predict(test[X])))})
    model = boosted().fit(d[X], d.mw)
    sample = d.sample(20000, random_state=0)
    imp = permutation_importance(model, sample[X], sample.mw, scoring="neg_mean_absolute_error", n_repeats=5,
                                 random_state=0)
    pdp = {k: partial_dependence(model, sample[X], [k], grid_resolution=40, percentiles=(0.02, 0.98))
           for k in ("bc_water", "bpat_demand", "hour", "temperature")}
    return pd.DataFrame(errors).set_index("year"), pd.Series(imp.importances_mean, index=X).sort_values(), pdp


def site_data(f, label, fraser):
    """The data behind the write-up's interactive charts, as a script (the site allows its own scripts but not
    fetching data files): weekly average flow by hour, each day's kind and average flow, and the Fraser's years."""
    start = balance.hour_start(f)
    heat = f.mw.groupby([start.dt.to_period("W-SUN").dt.start_time, start.dt.hour]).mean().unstack()
    heat = heat.dropna(thresh=20).round(0)
    p = daily_profiles(f)
    days = pd.date_range(p.index.min(), p.index.max())
    kind, mean = label.reindex(days), p.mean(axis=1).reindex(days).round(0)
    data = {"heat": {"weeks": [w.strftime("%Y-%m-%d") for w in heat.index],
                     "mw": [[None if pd.isna(v) else int(v) for v in row] for row in heat.to_numpy()]},
            "calendar": {"start": days[0].strftime("%Y-%m-%d"), "names": NAMES,
                         "pattern": "".join("-" if pd.isna(v) else str(int(v)) for v in kind),
                         "mw": [None if pd.isna(v) else int(v) for v in mean]},
            "river": {"years": [int(y) for y in fraser.index], "flow": [round(float(v)) for v in fraser]}}
    return "const BALANCE = " + json.dumps(data, separators=(",", ":")) + ";\n"


def main():
    f, b = balance.flows(), balance.balance()
    temperature = balance.temperature("1108395").combine_first(balance.temperature("1108447"))  # Vancouver
    p = daily_profiles(f)
    silhouette = {k: silhouette_score(p.values, KMeans(k, n_init=10, random_state=0).fit_predict(p.values),
                                      sample_size=3000, random_state=0) for k in range(2, 8)}
    pca = PCA(3).fit(p.values)
    label, centres = patterns(p)
    share = pd.crosstab(label.index.year, label, normalize="index").set_axis(NAMES, axis=1).rename_axis("year")
    anomaly, prob, fit, params, heavy = regimes(f)
    runs = spells(prob)
    d = frame(f, bpat(), b, temperature)
    errors, held = forecast(d)
    overall = weighted(errors)
    driver_errors, imp, pdp = drivers(d)
    driver_overall = weighted(driver_errors)

    share.round(3).to_csv(balance.OUTPUT / "patterns_by_year.csv")
    pd.DataFrame(centres, index=NAMES).round(0).rename_axis("pattern").to_csv(balance.OUTPUT / "pattern_profiles.csv")
    pd.DataFrame({"flow_vs_usual_mw": anomaly.round(0), "p_heavy_import": prob.round(3)}).rename_axis(
        "week_starting").to_csv(balance.OUTPUT / "import_regime.csv")
    runs.to_csv(balance.OUTPUT / "import_spells.csv", index=False)
    errors.round(1).to_csv(balance.OUTPUT / "flow_forecast_errors.csv")
    imp.round(1).rename("extra_error_mw").rename_axis("input").to_csv(balance.OUTPUT / "flow_drivers.csv")
    (balance.OUTPUT / "balance-data.js").write_text(site_data(f, label, rivers.annual_mean(rivers.river(rivers.FRASER))),
                                                    encoding="utf-8")
    print(f"{len(p)} days, {p.index.min():%Y-%m-%d} to {p.index.max():%Y-%m-%d}")
    print("Silhouette by k:", {k: round(v, 3) for k, v in silhouette.items()},
          "| PCA share of variance:", pca.explained_variance_ratio_.round(3))
    for i, name in enumerate(NAMES):
        c = centres[i]
        print(f"  {name}: {(label == i).sum()} days; mean {c.mean():.0f} MW, overnight {c[0:6].mean():.0f}, "
              f"10 am-4 pm {c[10:16].mean():.0f}, 5-9 pm {c[17:21].mean():.0f}")
    print((share * 100).round(0).to_string())
    print(params.round(3).to_string(), "\nExpected weeks in each regime:", np.round(fit.expected_durations, 1))
    print(runs.to_string())
    print(f"{len(d):,} model hours, {d.start.min():%Y-%m-%d} to {d.start.max():%Y-%m-%d}")
    print("Day-ahead mean absolute error (MW), trained on earlier years:\n", errors.round(0).to_string())
    print("Overall:", overall.round(0).to_dict())
    print("Same-hour model, each year held out:", driver_overall.round(0).to_dict())
    print("Permutation importance (MW of extra error):", imp.round(0).to_dict())
    for k, r in pdp.items():
        g, a = r["grid_values"][0], r["average"][0]
        print(f"  {k}: {g[0]:.0f} -> {a[0]:.0f} MW, {g[-1]:.0f} -> {a[-1]:.0f} MW")

    plt.rcParams.update(balance.STYLE)
    eia = "Source: US Energy Information Administration, Form EIA-930 (BPAT-BCHA interchange)."
    # Every hour since 2017
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(top=0.8, bottom=0.17, right=0.93)
    lim = 2500
    img = ax.imshow(p.T.values, aspect="auto", cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -lim, lim), interpolation="none",
                    extent=[mdates.date2num(p.index[0]), mdates.date2num(p.index[-1] + pd.Timedelta(days=1)), 24, 0])
    ax.xaxis_date()
    ax.set_yticks([0, 6, 12, 18, 24], ["midnight", "6 am", "noon", "6 pm", "midnight"])
    ax.grid(False)
    bar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("MW into BC (blue: out of BC)", color=MUTED)
    bar.outline.set_visible(False)
    balance.save(fig, "hourly_heatmap.png",
                 "Every hour since 2017: the drought shows as two years of almost solid importing",
                 "Hourly flow between the Bonneville Power Administration and BC Hydro, each day (across) by Pacific hour (down)",
                 f"Colour capped at {lim:,} MW either way. Spring clock-change days left out.\n" + eia)

    # The kinds of day and when each occurred
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(top=0.79, bottom=0.25, wspace=0.18)
    for i, name in enumerate(NAMES):
        days = p[label == i]
        left.fill_between(range(24), days.quantile(0.1), days.quantile(0.9), color=COLOURS[i], alpha=0.15, lw=0)
        left.plot(range(24), centres[i], color=COLOURS[i], lw=2.4, label=f"{name} ({len(days):,} days)")
    left.axhline(0, color=MUTED, lw=0.8)
    left.set_xticks([0, 6, 12, 18, 23], ["midnight", "6 am", "noon", "6 pm", "11 pm"])
    left.set_ylabel("MW into BC")
    left.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=9)
    left.set_title("The three typical days (bands: middle 80%)", loc="left", color=INK, fontsize=12)
    years = range(p.index.year.min(), p.index.year.max() + 1)
    grid = np.full((len(years), 366), np.nan)
    for day, g in label.items():
        grid[day.year - years[0], day.dayofyear - 1] = g
    right.imshow(grid, aspect="auto", cmap=mcolors.ListedColormap(COLOURS), vmin=-0.5, vmax=K - 0.5, interpolation="none",
                 extent=[0.5, 366.5, years[-1] + 0.5, years[0] - 0.5])
    firsts = pd.date_range("2025-01-01", periods=12, freq="MS")
    right.set_xticks(firsts.dayofyear, [m.strftime("%b") for m in firsts])
    right.set_yticks(list(years))
    right.grid(False)
    right.set_title("Each day's pattern", loc="left", color=INK, fontsize=12)
    peak = share[NAMES[2]].idxmax()
    balance.save(fig, "trading_patterns.png",
                 f"Three kinds of trading day, and the drought made {share.loc[peak, NAMES[2]]:.0%} of {peak}'s days "
                 "importing all day",
                 "Days grouped by the shape of their 24-hour flow with k-means; each group's average day and the calendar "
                 "of which days fell in it",
                 f"k = 3: silhouette peaks at k = 2 ({silhouette[2]:.2f}, import vs export); a third group "
                 f"({silhouette[3]:.2f}) splits off the two-way days. White: spring clock-change days.\n" + eia)

    # The heavy-import regime
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.8, bottom=0.15, hspace=0.12)
    top.bar(anomaly.index, anomaly, width=6, color=[RED if v > 0 else BLUE for v in anomaly], lw=0)
    top.axhline(0, color=MUTED, lw=0.8)
    top.set_ylabel("MW vs usual")
    top.set_title(f"Weekly flow into BC, against the {balance.FLOWS_BASELINE[0]}-{balance.FLOWS_BASELINE[1]} average "
                  "for the month", loc="left", color=INK, fontsize=12)
    bottom.fill_between(prob.index, prob, color=RED, alpha=0.35, lw=0, step="mid")
    bottom.plot(prob.index, prob, color=RED, lw=1.2, drawstyle="steps-mid")
    bottom.set_ylim(0, 1.05)
    bottom.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    bottom.set_ylabel("Chance in\nimport regime")
    for _, r in runs.iterrows():
        for ax in (top, bottom):
            ax.axvspan(r.start, r.end, color=GREY, alpha=0.12, lw=0)
    longest = runs.loc[runs.weeks.idxmax()]
    means = params.filter(like="const").round(-1)
    balance.save(fig, "import_regime.png",
                 f"A regime model dates the drought's heavy importing to {longest.start:%B %Y}-{longest.end:%B %Y}",
                 "Two-regime Markov-switching model of weekly flows; shaded: spells of four weeks or more in which the "
                 "heavy-import regime is the likelier",
                 f"Regime means about {means.iloc[heavy]:+.0f} and {means.iloc[1 - heavy]:+.0f} MW against the usual; each "
                 f"lasts about {np.mean(fit.expected_durations):.0f} weeks on average.\n" + eia)

    sources = ("Sources: US EIA Form EIA-930 (BPAT-BCHA flows; BPAT demand, demand forecast, generation); Statistics "
               "Canada table 25-10-0015-01; Government of Canada climate data.")
    # The day-ahead forecast
    names, colours = ["Usual for the month and hour", "Same hour yesterday", "Gradient boosting", "Linear (ridge)"], \
        [GREY, ORANGE, RED, BLUE]
    gain = 1 - overall["Linear (ridge)"] / overall["Same hour yesterday"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.27, wspace=0.2)
    window = held[held.start >= held.start.max().normalize() - pd.Timedelta(days=14)]
    left.plot(window.start, window.actual, color=INK, lw=1.8, label="Actual")
    left.plot(window.start, window["Same hour yesterday"], color=ORANGE, lw=1.2, label="Same hour yesterday")
    left.plot(window.start, window["Linear (ridge)"], color=BLUE, lw=1.6, label="Linear forecast")
    left.axhline(0, color=MUTED, lw=0.8)
    left.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    left.set_ylabel("MW into BC")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    left.set_title(f"The last two weeks, forecast each midnight ({window.start.min():%d %b} to "
                   f"{window.start.max():%d %b %Y})", loc="left", color=INK, fontsize=12)
    x, w = np.arange(len(errors)), 0.2
    for i, (name, colour) in enumerate(zip(names, colours)):
        right.bar(x + (i - 1.5) * w, errors[name], w, color=colour, label=f"{name}: {overall[name]:.0f} MW")
    right.set_xticks(x, [str(y) + ("\nso far" if y == errors.index.max() else "") for y in errors.index], fontsize=9)
    right.set_ylabel("Average error, MW")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    right.set_title("Error each year, trained on earlier years", loc="left", color=INK, fontsize=12)
    worse = [y for y in errors.index if errors.loc[y, "Linear (ridge)"] > errors.loc[y, "Same hour yesterday"]]
    balance.save(fig, "flow_forecast.png",
                 f"A day-ahead forecast of the hourly flow beats repeating yesterday by {gain:.0%}"
                 + (f", except in {' and '.join(map(str, worse))}" if worse else ""),
                 "Forecasts made at midnight for the next 24 hours; each year forecast by models trained only on the "
                 "years before",
                 "Inputs: yesterday's flows and Bonneville output, Bonneville's demand forecast, Vancouver's actual "
                 "temperature (in place of a forecast), BC's hydro over the past year, calendar.\n" + sources)

    # What drives the flow
    cut = 1 - driver_overall["boosting"] / driver_overall["usual"]
    fig = plt.figure(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.17, wspace=0.38, hspace=0.6, left=0.2)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.barh([DRIVERS[k] for k in imp.index], imp, color=BLUE)
    ax.set_xlabel("Extra error when shuffled, MW")
    ax.set_title("What the model relies on", loc="left", color=INK, fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    units = {"bc_water": "% of 2013-2022 average", "bpat_demand": "MW", "hour": "Pacific hour beginning",
             "temperature": "°C"}
    for i, k in enumerate(pdp):
        a = fig.add_subplot(gs[i // 2, 1 + i % 2])
        a.plot(pdp[k]["grid_values"][0], pdp[k]["average"][0], color=RED, lw=2.2)
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_title(DRIVERS[k], loc="left", color=INK, fontsize=11)
        a.set_xlabel(units[k], fontsize=9)
        a.tick_params(labelsize=9)
        if i % 2 == 0:
            a.set_ylabel("MW into BC", fontsize=9)
    balance.save(fig, "flow_drivers.png",
                 "BC's own water decides the direction of trade more than any condition across the border",
                 "Gradient boosting on same-hour conditions. Right: partial dependence, the model's average flow as one "
                 "input changes and the rest stay as observed",
                 f"Importance on the fitted hours. Held out by year, the model beats the usual for the month and hour by "
                 f"only {cut:.0%}; prices and reservoir plans are not in open data.\n" + sources)


if __name__ == "__main__":
    main()
