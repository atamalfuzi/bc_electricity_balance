"""Monthly dashboard for British Columbia's electricity balance: the latest month's demand against what the
trend, season and weather predicted, the last 12 months against the 12 before (weather-adjusted), hydro output
against normal, net imports, what trade with the United States earned or cost, and the last full month of hourly
flows with the United States. Writes a static site to site/.

Run from the repository folder, after the analyses (its longer view uses their charts):
python balance.py && python rivers.py && python trading.py && python forecast.py && python dashboard.py
"""
from pathlib import Path
import datetime as dt
import json
import shutil
import urllib.error
import urllib.request

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import balance
from balance import BLUE, GREEN, GREY, INK, MUTED, ORANGE, RED

SITE = Path("site")
URL = "https://atamalfuzi.github.io/bc_electricity_balance/"
WRITE_UP = "https://atamalfuzi.github.io/bc-electricity-balance.html"


def series():
    """The monthly balance, weather and trade, cut to the last month all three cover, with the load expected
    by the state-space model from earlier months and the weather-normalised load."""
    b, t = balance.balance(), balance.trade()
    d = balance.monthly(b, balance.weather())
    end = min(b.dropna().index.max(), t.dropna().index.max(), d.dropna(subset=["hdd", "cdd"]).index.max())
    b, t, d = b.loc[:end], t.loc[:end], d.loc[:end]
    _, kalman = balance.temperature_effect(d)
    expected = pd.Series(kalman.fittedvalues, index=d.index)  # one month ahead, given that month's weather
    normal = balance.normalised(d, balance.demand_model(d))
    return b, t, d, expected, normal, balance.flows()


def gwh(mw):
    """Monthly average MW to GWh in the month."""
    return mw * 24 * mw.index.days_in_month / 1000


def summary(b, t, expected_mw, normal_mw):
    last = b.index.max()
    year, before = slice(last - 11, last), slice(last - 23, last - 12)
    base = b.loc[f"{balance.BASELINE[0]}-01":f"{balance.BASELINE[1]}-12"]
    trade = t.loc[year].sum()
    return {
        "month": str(last),
        "demand_gwh": round(float(b.demand[last])),
        "expected_gwh": round(float(gwh(expected_mw)[last])),
        "demand_vs_expected": round(float(b.demand[last] / gwh(expected_mw)[last] - 1), 4),
        "demand_last_12_vs_previous_12": round(float(gwh(normal_mw)[year].sum() / gwh(normal_mw)[before].sum() - 1), 4),
        # 12 consecutive months cover each calendar month once, so the normal is the baseline's annual average
        "hydro_last_12_vs_normal": round(float(b.hydro[year].sum() / (base.hydro.sum() / len(base) * 12) - 1), 4),
        "net_imports_last_12_twh": round(float(b.net_imports[year].sum() / 1000), 2),
        "net_imports_previous_12_twh": round(float(b.net_imports[before].sum() / 1000), 2),
        "net_imports_share_of_demand": round(float(b.net_imports[year].sum() / b.demand[year].sum()), 4),
        "us_trade_net_last_12_million": round(float(trade.export_value - trade.import_value)),
        "export_price_last_12": round(float(trade.export_value / trade.export_gwh * 1000), 1),
        "import_price_last_12": round(float(trade.import_value / trade.import_gwh * 1000), 1),
    }


def last_full_month(f):
    """The latest calendar month the hourly flows cover to its end."""
    return (balance.hour_start(f).max() + pd.Timedelta(hours=1)).to_period("M") - 1


def hourly(f):
    """The latest full month of hourly flows: the share of hours BC was importing (and in the same month of the
    baseline years), and the average flow into BC at the evening peak and overnight (negative: out of BC)."""
    month = last_full_month(f)
    start = balance.hour_start(f)
    this = start.dt.to_period("M") == month
    base = (start.dt.month == month.month) & start.dt.year.between(*balance.FLOWS_BASELINE)
    hour = start.dt.hour
    return {
        "hourly_month": str(month),
        "hourly_share_importing": round(float((f.mw[this] > 0).mean()), 4),
        "hourly_share_importing_baseline": round(float((f.mw[base] > 0).mean()), 4),
        "hourly_evening_mw": round(float(f.mw[this & hour.isin(balance.EVENING)].mean())),
        "hourly_overnight_mw": round(float(f.mw[this & hour.isin(balance.OVERNIGHT)].mean())),
        "hourly_net_gwh": round(float(f.mw[this].sum() / 1000)),
    }


# Checks before publishing ----
def problems(b, t, d, f, published_through=None):
    """Reasons not to publish: gaps or implausible values in the last two years or the last full month of hourly
    flows, or data older than what is live."""
    last = b.index.max()
    months = pd.period_range(last - 23, last, freq="M")
    recent = b.reindex(months).join(t.reindex(months)).join(d[["hdd", "cdd"]].reindex(months))
    found = []
    if recent.isna().any().any():
        found.append("missing values in the last 24 months")
    if not recent.demand.dropna().between(3000, 9000).all():
        found.append("demand outside 3,000 to 9,000 GWh a month")
    if not recent.hydro.dropna().between(1500, 9000).all():
        found.append("hydro output outside 1,500 to 9,000 GWh a month")
    prices = (recent[["export_value", "import_value"]].to_numpy() / recent[["export_gwh", "import_gwh"]].to_numpy() * 1000)
    prices = prices[~np.isnan(prices)]
    if not ((prices > 0) & (prices < 1000)).all():
        found.append("trade prices outside $0 to $1,000 per MWh")
    month = last_full_month(f)
    hours = f.mw[balance.hour_start(f).dt.to_period("M") == month].dropna()
    if len(hours) < 0.95 * month.days_in_month * 24:
        found.append(f"hourly flows missing for more than 5% of {month}")
    if not hours.abs().le(4000).all():
        found.append("hourly flows beyond 4,000 MW either way")
    if published_through and str(last) < published_through:
        found.append(f"data end {last} is older than the published {published_through}")
    return found


def published_through():
    """The data month of the dashboard now live, if there is one."""
    try:
        with urllib.request.urlopen(URL + "status.json", timeout=30) as response:
            return json.load(response)["data_through"]
    except (urllib.error.URLError, KeyError, ValueError):  # first run, or no live copy to compare with
        return None


# Figures ----
SOURCE = "Source: Statistics Canada tables 25-10-0015-01 and 25-10-0016-01"


def years_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def figures(b, expected_mw, d, f):
    plt.rcParams.update(balance.STYLE)
    recent = slice(b.index.max() - 35, b.index.max())
    months = b.loc[recent].index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.8, bottom=0.2)
    ax.plot(months, gwh(d.load)[recent], color=BLUE, lw=2.2, label="Actual")
    ax.plot(months, gwh(expected_mw)[recent], color=MUTED, lw=1.5, ls="--", label="Expected")
    ax.set_ylabel("GWh a month")
    ax.legend(loc="upper right")
    years_axis(ax)
    balance.save(fig, "recent_demand.png", "BC's electricity demand, actual and expected, last three years",
                 "Expected: what the trend, the season and the month's weather predicted from the months before",
                 "State-space model of monthly average load with heating and cooling degree days at four airports.\n"
                 + SOURCE + "; Government of Canada daily climate data.", folder=SITE, dpi=150)

    base = b.loc[f"{balance.BASELINE[0]}-01":f"{balance.BASELINE[1]}-12"].hydro
    by_month = base.groupby(base.index.month)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(top=0.79, bottom=0.25, wspace=0.22)
    calendar = b.loc[recent].index.month
    left.fill_between(months, by_month.min().reindex(calendar), by_month.max().reindex(calendar), color=GREY,
                      alpha=0.3, lw=0, label=f"{balance.BASELINE[0]}-{balance.BASELINE[1]} range")
    left.plot(months, by_month.mean().reindex(calendar), color=MUTED, lw=1.5, ls="--", label="Average")
    left.plot(months, b.hydro[recent], color=BLUE, lw=2.2, label="Hydro output")
    left.set_ylabel("GWh a month")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=3)
    left.set_title("Hydro generation against its usual range", loc="left", color=INK, fontsize=12)
    net = b.net_imports[recent]
    right.bar(months, net, width=25, color=[RED if v > 0 else GREEN for v in net])
    right.axhline(0, color=MUTED, lw=0.8)
    right.set_ylabel("GWh a month")
    right.set_title("Net imports (red) and net exports (green)", loc="left", color=INK, fontsize=12)
    for ax in (left, right):
        years_axis(ax)
    balance.save(fig, "recent_supply.png", "Hydro output and net imports, last three years",
                 "Net imports are receipts from other provinces and the US minus deliveries to them", SOURCE + ".",
                 folder=SITE, dpi=150)

    month = last_full_month(f)
    start = balance.hour_start(f)
    hour, period = start.dt.hour, start.dt.to_period("M")
    base = (start.dt.month == month.month) & start.dt.year.between(*balance.FLOWS_BASELINE)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.subplots_adjust(top=0.8, bottom=0.27)
    ax.axvspan(balance.EVENING[0] - 0.5, balance.EVENING[-1] + 0.5, color=GREY, alpha=0.15, lw=0)
    baseline = f"{month.strftime('%B')} {balance.FLOWS_BASELINE[0]}-{balance.FLOWS_BASELINE[1]} average"
    for label, mask, colour, style, width in [(baseline, base, MUTED, "--", 1.6),
                                              ((month - 12).strftime("%B %Y"), period == month - 12, ORANGE, "-", 1.8),
                                              (month.strftime("%B %Y"), period == month, BLUE, "-", 2.6)]:
        ax.plot(range(24), f.mw[mask].groupby(hour[mask]).mean().reindex(range(24)), color=colour, ls=style, lw=width,
                label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 6, 12, 18, 23], ["midnight", "6 am", "noon", "6 pm", "11 pm"])
    ax.set_ylabel("MW into BC (negative: out of BC)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=3)
    balance.save(fig, "recent_hourly.png", f"Power flow with the US by hour of the day, {month.strftime('%B %Y')}",
                 "Average flow into BC by Pacific hour, against the same month in earlier years (5-9 pm shaded)",
                 "Flow between the Bonneville Power Administration and BC Hydro.\n"
                 f"Source: US Energy Information Administration, Form EIA-930, {dt.date.today():%B %Y}.",
                 folder=SITE, dpi=150)


# Page ----
def pct(x):
    return f"{x:+.1%}".replace("-", "−")


def money(x):
    return f"{'+' if x >= 0 else '−'}${abs(x):,.0f} million"


def mw(x):
    return f"{abs(x):,} MW {'in' if x >= 0 else 'out'}"


def flow(twh):
    """Net imports as 'net imports' or 'net exports' and a positive amount."""
    return ("net imports" if twh >= 0 else "net exports"), abs(twh)


def page(s, data_through, updated):
    month = pd.Period(data_through, "M").strftime("%B %Y")
    hourly_month = pd.Period(s["hourly_month"], "M")
    now, now_twh = flow(s["net_imports_last_12_twh"])
    before, before_twh = flow(s["net_imports_previous_12_twh"])
    cards = f"""
      <section class="card">
        <h2>Demand</h2>
        <dl>
          <div><dt>{month} vs expected</dt><dd>{pct(s['demand_vs_expected'])}</dd>
            <p>{s['demand_gwh']:,} GWh, against {s['expected_gwh']:,} expected</p></div>
          <div><dt>Last 12 months vs the 12 before</dt><dd>{pct(s['demand_last_12_vs_previous_12'])}</dd>
            <p>Weather-adjusted to an average year</p></div>
        </dl>
      </section>
      <section class="card">
        <h2>Hydro and imports</h2>
        <dl>
          <div><dt>Hydro output, last 12 months</dt><dd>{pct(s['hydro_last_12_vs_normal'])}</dd>
            <p>Against the {balance.BASELINE[0]}-{balance.BASELINE[1]} average</p></div>
          <div><dt>{now.capitalize()}, last 12 months</dt><dd>{now_twh:.1f} TWh</dd>
            <p>Equal to {abs(s['net_imports_share_of_demand']):.0%} of demand; {before} of {before_twh:.1f} TWh the 12 months before</p></div>
        </dl>
      </section>
      <section class="card">
        <h2>Trade with the US</h2>
        <dl>
          <div><dt>Exports minus imports, last 12 months</dt><dd>{money(s['us_trade_net_last_12_million'])}</dd>
            <p>Value of power sold less value bought</p></div>
          <div><dt>Price received vs paid</dt><dd>${s['export_price_last_12']:.0f} / ${s['import_price_last_12']:.0f}</dd>
            <p>Per MWh, exports and imports, last 12 months</p></div>
        </dl>
      </section>
      <section class="card">
        <h2>Trade by the hour</h2>
        <dl>
          <div><dt>Hours importing, {hourly_month.strftime('%B %Y')}</dt><dd>{s['hourly_share_importing']:.0%}</dd>
            <p>Against {s['hourly_share_importing_baseline']:.0%} in {hourly_month.strftime('%B')} {balance.FLOWS_BASELINE[0]}-{balance.FLOWS_BASELINE[1]}</p></div>
          <div><dt>Evening peak, 5-9 pm</dt><dd>{mw(s['hourly_evening_mw'])}</dd>
            <p>Average flow; {mw(s['hourly_overnight_mw'])} overnight (midnight to 6 am)</p></div>
        </dl>
      </section>"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'self'; form-action 'none'" />
  <title>BC electricity balance monitor | Ata Malfuzi</title>
  <meta name="description" content="Monthly update on British Columbia's electricity demand, hydro output, net imports and trade with the United States, from open data." />
  <link rel="canonical" href="{URL}" />
  <link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <p class="eyebrow"><a href="https://atamalfuzi.github.io/">Ata Malfuzi</a> · Monthly dashboard</p>
  <h1>BC electricity balance monitor</h1>
  <p class="meta">Data through {month} · updated {updated:%d %B %Y} · refreshes on the 15th of each month</p>
  <p class="links"><a href="{WRITE_UP}">Full write-up</a><a href="https://github.com/atamalfuzi/bc_electricity_balance">Code on GitHub</a><a href="status.json">Latest numbers (JSON)</a></p>
  <div class="cards">{cards}
  </div>
  <figure><img src="recent_demand.png" alt="BC's monthly electricity demand over the last three years, actual against expected." width="1800" height="750" /></figure>
  <figure><img src="recent_supply.png" alt="Two panels: BC's monthly hydro output over the last three years against its 2013-2022 range, and monthly net imports." width="1800" height="810" /></figure>
  <figure><img src="recent_hourly.png" alt="BC's average power flow with the United States by hour of the day in the latest full month, against the same month a year earlier and in 2018-2022." width="1800" height="810" /></figure>
  <h2>The longer view</h2>
  <p>Re-estimated each month with all data to date.</p>
  <figure><img src="demand_forecast.png" alt="BC's monthly demand over the last three years and the forecast for the next six months with 80% and 95% ranges, and the average error of three forecasting methods by months ahead." width="2400" height="1120" /></figure>
  <figure><img src="import_regime.png" alt="Weekly power flow into BC against the usual for the month since 2017, and the probability from a regime model that each week was in a heavy-import spell." width="2400" height="1160" /></figure>
  <figure><img src="flow_forecast.png" alt="The last two weeks of hourly power flow with the United States against a day-ahead forecast, and each forecasting method's average error by year." width="2400" height="1160" /></figure>
  <figure><img src="drought.png" alt="Annual generation and demand in BC since 1990 with net imports and exports shaded, and the 2023-2025 rise in net imports split into lower hydro output, demand growth, weather and other generation." width="2400" height="1200" /></figure>
  <figure><img src="trade.png" alt="Value of BC's electricity exports to and imports from the United States since 1990, and the average prices received and paid." width="2400" height="1040" /></figure>
  <figure><img src="hourly_flows.png" alt="Average hourly power flow between BC and the United States by hour of the day for 2018-2022 and each later year, and the share of hours BC was importing each year since 2018." width="2400" height="1160" /></figure>
  <figure><img src="hydro_capacity.png" alt="BC's hydro output as a share of installed hydro capacity each year since 2006, lowest in 2023, and installed capacity by type." width="2400" height="1040" /></figure>
  <figure><img src="temperature_effect.png" alt="Effect of a degree colder and a degree hotter on BC's electricity demand, estimated year by year since 1980 with 90% bands." width="2400" height="1040" /></figure>
  <footer>
    <p>Personal project: my own analysis of publicly available data. It is not associated with my work for any current or past employer and does not represent their views.</p>
    <p>Adapted from Statistics Canada, tables 25-10-0001-01, 25-10-0015-01 and 25-10-0016-01; this does not constitute an endorsement by Statistics Canada of this product. Trade data from the Canada Energy Regulator. Contains information licensed under the Open Government Licence - Canada. Hourly flows from the US Energy Information Administration (Form EIA-930).</p>
  </footer>
</main>
</body>
</html>
"""


CSS = """:root { --text: #17201d; --muted: #4b5651; --rule: #e1e7e4; --accent: #0b6e5f; --bg-2: #f4f6f5; }
* { box-sizing: border-box; margin: 0; }
body { font-family: system-ui, sans-serif; color: var(--text); background: #fff; line-height: 1.55; }
main { max-width: 1000px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
h1 { font-family: Georgia, serif; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.2; margin: .3rem 0 .5rem; }
h2 { font-family: Georgia, serif; font-size: 1.3rem; margin: 2rem 0 .5rem; }
a { color: var(--accent); }
.eyebrow, .meta, footer { color: var(--muted); font-size: .9rem; }
.links { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin: .8rem 0 1.5rem; font-weight: 600; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); gap: 1rem; }
.card { border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.2rem; background: var(--bg-2); }
.card h2 { margin: 0 0 .6rem; font-size: 1.15rem; }
.card dl { display: grid; gap: .8rem; }
dt { font-size: .85rem; color: var(--muted); }
dd { font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.card p { font-size: .8rem; color: var(--muted); }
figure { margin: 1.5rem 0; }
img { width: 100%; height: auto; display: block; }
footer { border-top: 1px solid var(--rule); margin-top: 2rem; padding-top: 1rem; display: grid; gap: .5rem; }
"""


def main():
    SITE.mkdir(exist_ok=True)
    b, t, d, expected, normal, f = series()
    issues = problems(b, t, d, f, published_through())
    if issues:
        raise SystemExit("Not publishing: " + "; ".join(issues))

    s = {**summary(b, t, expected, normal), **hourly(f)}
    figures(b, expected, d, f)
    for name in ("demand_forecast.png", "import_regime.png", "flow_forecast.png", "drought.png", "trade.png",
                 "hourly_flows.png", "hydro_capacity.png", "temperature_effect.png"):
        shutil.copy(balance.FIGURES / name, SITE / name)
    updated = dt.date.today()
    (SITE / "index.html").write_text(page(s, s["month"], updated), encoding="utf-8")
    (SITE / "style.css").write_text(CSS, encoding="utf-8")
    status = {"data_through": s["month"], "updated": updated.isoformat(), **s}
    (SITE / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
