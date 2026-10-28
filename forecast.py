"""Forecasts with uncertainty. Monthly demand one to six months ahead from three methods, the same month last year,
exponential smoothing and a seasonal ARIMA with heating and cooling degree days (normal weather assumed ahead), each
tested from every month since 2015 using only the data before it. Prediction ranges come from the model's own past
errors (conformal prediction), and their coverage is checked as it would have run. Also 90% ranges for the weather
adjustment of each year's demand.

Run from the repository folder, after balance.py: python forecast.py (about five minutes)
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import balance
from balance import BLUE, GREY, INK, MUTED, ORANGE, RED, gwh

FIT_FROM = "2000-01"
FIRST_ORIGIN = "2014-12"
H = 6
BEST = "Seasonal ARIMA with weather"
SOURCE = ("Source: Statistics Canada tables 25-10-0001-01, 25-10-0015-01 and 25-10-0016-01; Government of Canada "
          "daily climate data.")


def models(y, X, future_X, steps=H):
    """Each method's point forecasts (MW) for the months after y, with the models' own 80% and 95% ranges."""
    months = pd.period_range(y.index[-1] + 1, periods=steps, freq="M")
    s = SARIMAX(y, exog=X, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
    f = s.get_forecast(steps, exog=future_X)
    arima = pd.DataFrame({"mean": f.predicted_mean.to_numpy(), **{
        f"{q}_{a}": f.conf_int(alpha=1 - a / 100).iloc[:, i].to_numpy() for a in (80, 95) for i, q in enumerate(("low", "high"))}})
    e = ETSModel(y, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit(disp=False)
    p = e.get_prediction(start=len(y), end=len(y) + steps - 1)
    ets = pd.DataFrame({"mean": p.predicted_mean.to_numpy(), **{
        f"{q}_{a}": p.summary_frame(alpha=1 - a / 100)[f"pi_{q2}er"].to_numpy()
        for a in (80, 95) for q, q2 in (("low", "low"), ("high", "upp"))}})
    last_year = pd.DataFrame({"mean": y.reindex(months - 12).to_numpy()})
    return {k: v.set_axis(months) for k, v in
            {BEST: arima, "Exponential smoothing": ets, "Same month last year": last_year}.items()}


def normal_weather(d, months):
    """The 1991-2020 average heating and cooling degrees per day for each month."""
    normal = d.groupby(d.index.month)[["hdd_normal", "cdd_normal"]].first().set_axis(["hdd", "cdd"], axis=1)
    return normal.reindex(months.month).set_axis(months)


def backtest(d):
    """Forecasts from every month since FIRST_ORIGIN, each fitted to the data before it, against what happened."""
    y_all = d.load.loc[FIT_FROM:]
    rows = []
    for origin in pd.period_range(FIRST_ORIGIN, y_all.index[-1] - 1, freq="M"):
        y = y_all.loc[:origin]
        ahead = pd.period_range(origin + 1, periods=H, freq="M")
        for name, f in models(y, d.loc[y.index, ["hdd", "cdd"]], normal_weather(d, ahead)).items():
            for step, (month, r) in enumerate(f.iterrows(), start=1):
                if month in y_all.index:
                    rows.append({"origin": origin, "month": month, "horizon": step, "model": name,
                                 "actual": y_all[month], **r.to_dict()})
    return pd.DataFrame(rows)


def conformal(bt, model, levels=(80, 95), warmup=24):
    """Ranges from the model's own past errors: at each horizon, percentiles of the absolute % errors. Coverage is
    checked as it would have run, from only the errors known at each forecast's origin (after `warmup` of them);
    the widths returned use all of them."""
    e = bt[bt.model == model].assign(abs_err=lambda t: (t.actual - t["mean"]).abs() / t["mean"])
    hits = {a: [] for a in levels}
    for _, r in e.iterrows():
        past = e[(e.horizon == r.horizon) & (e.month <= r.origin)].abs_err
        if len(past) >= warmup:
            for a in levels:
                hits[a].append(r.abs_err <= past.quantile(a / 100))
    widths = e.groupby("horizon").abs_err.quantile([a / 100 for a in levels]).unstack()
    return {a: float(np.mean(v)) for a, v in hits.items()}, widths.set_axis(list(levels), axis=1)


def weather_ranges(d, fit):
    """Each full year's weather effect (actual minus weather-normalised demand, TWh) with a 90% range from the
    demand model's Newey-West covariance."""
    cols = balance.weather_columns(d).columns
    gap = balance.weather_columns(d) - balance.weather_columns(d.assign(hdd=d.hdd_normal, cdd=d.cdd_normal))
    gap = gap.mul(24 * d.index.days_in_month / 1e6, axis=0)  # MW for a month to TWh
    cov = fit.cov_params().loc[cols, cols]
    rows = {}
    for year, g in gap.groupby(d.index.year):
        if g.notna().all(axis=1).sum() == 12:
            w = g.sum()
            effect, se = float(w @ fit.params[cols]), float(np.sqrt(w @ cov @ w))
            rows[year] = {"weather_twh": effect, "low_90": effect - 1.645 * se, "high_90": effect + 1.645 * se}
    return pd.DataFrame(rows).T.rename_axis("year")


def main():
    d = balance.monthly(balance.balance(), balance.weather())
    d = d.loc[:d.dropna(subset=["load", "hdd", "cdd"]).index.max()]
    bt = backtest(d)
    bt["pct_error"] = (bt["mean"] - bt.actual).abs() / bt.actual
    mape = bt.pivot_table(index="horizon", columns="model", values="pct_error", aggfunc="mean") * 100
    own = {a: bt.dropna(subset=[f"low_{a}"]).assign(inside=lambda t: t.actual.between(t[f"low_{a}"], t[f"high_{a}"]))
           .groupby("model").inside.mean() for a in (80, 95)}
    cover, widths = conformal(bt, BEST)
    y = d.load.loc[FIT_FROM:]
    ahead = pd.period_range(y.index[-1] + 1, periods=H, freq="M")
    mean = gwh(models(y, d.loc[y.index, ["hdd", "cdd"]], normal_weather(d, ahead))[BEST]["mean"])
    fan = pd.DataFrame({"mean": mean, **{f"{q}_{a}": mean * (1 + s * widths[a].to_numpy())
                                         for a in (80, 95) for q, s in (("low", -1), ("high", 1))}})
    ranges = weather_ranges(d, balance.demand_model(d))

    mape.round(2).to_csv(balance.OUTPUT / "demand_forecast_errors.csv")
    fan.round(0).rename_axis("month").to_csv(balance.OUTPUT / "demand_forecast.csv")
    ranges.round(2).to_csv(balance.OUTPUT / "weather_effect.csv")
    print(f"{bt.origin.nunique()} forecast origins, {bt.origin.min()} to {bt.origin.max()}")
    print("Mean absolute % error by months ahead:\n", mape.round(2).to_string())
    print("The models' own ranges held:", {a: v.round(3).to_dict() for a, v in own.items()})
    print("Conformal ranges held (built from earlier errors only):", {a: round(v, 3) for a, v in cover.items()})
    print(fan.round(0).to_string())
    print(ranges.loc[2015:].round(2).to_string())

    plt.rcParams.update(balance.STYLE)
    # The forecast, and how good such forecasts have been
    names, colours = ["Same month last year", "Exponential smoothing", BEST], [GREY, ORANGE, BLUE]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.25, wspace=0.22)
    hist = gwh(y.iloc[-36:])
    x_hist, x_fan = hist.index.to_timestamp(), ahead.to_timestamp()
    left.plot(x_hist, hist, color=INK, lw=2, label="Actual")
    left.fill_between(x_fan, fan.low_95, fan.high_95, color=BLUE, alpha=0.15, lw=0, label="95% range")
    left.fill_between(x_fan, fan.low_80, fan.high_80, color=BLUE, alpha=0.3, lw=0, label="80% range")
    left.plot([x_hist[-1], *x_fan], [hist.iloc[-1], *fan["mean"]], color=BLUE, lw=2, ls="--", label="Forecast")
    left.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    left.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    left.set_ylabel("GWh a month")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=9)
    left.set_title(f"BC's demand, with the next six months ({ahead[0]:%b %Y} to {ahead[-1]:%b %Y})")
    for name, colour in zip(names, colours):
        right.plot(mape.index, mape[name], color=colour, lw=2.2, marker="o", label=name)
    right.set_xticks(range(1, H + 1))
    right.set_xlabel("Months ahead")
    right.set_ylabel("Average error, %")
    right.set_ylim(0, None)
    right.legend(loc="lower right", fontsize=9)
    right.set_title("How far off such forecasts have been")
    balance.save(fig, "demand_forecast.png",
                 f"Six months out, BC's demand can be forecast to within about {mape[BEST].iloc[-1]:.1f}% on average",
                 f"Seasonal ARIMA with heating and cooling degree days (normal weather assumed ahead); tested from every "
                 f"month since {bt.origin.min():%Y} using only earlier data",
                 f"Ranges from the model's past errors (conformal): built only from earlier errors they held {cover[80]:.0%} "
                 f"and {cover[95]:.0%} of outcomes; the model's own held {own[80][BEST]:.0%} and {own[95][BEST]:.0%}.\n"
                 + SOURCE)

    # The weather adjustment with its uncertainty
    r = ranges
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(top=0.8, bottom=0.17)
    ax.bar(r.index, r.weather_twh, color=[RED if v > 0 else BLUE for v in r.weather_twh], width=0.7,
           yerr=[r.weather_twh - r.low_90, r.high_90 - r.weather_twh], capsize=2, ecolor=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("TWh against normal weather")
    clear = (np.sign(r.low_90) == np.sign(r.high_90)).mean()
    balance.save(fig, "weather_effect.png",
                 f"Weather moves BC's demand by up to {r.weather_twh.abs().max():.1f} TWh a year, clear of zero in "
                 f"{clear:.0%} of years",
                 "Each year's demand minus its weather-normalised value (red: colder or hotter than normal added demand), "
                 "with 90% ranges",
                 "Ranges from the Newey-West covariance of the monthly demand model's heating and cooling effects.\n" + SOURCE)


if __name__ == "__main__":
    main()
