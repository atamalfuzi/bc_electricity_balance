"""British Columbia's electricity balance from open data: how much of the growth in demand since 1980 is
weather, how its response to temperature has changed, and where the power came from, and at what cost, when
the 2023-2024 drought cut hydro output.

Data: Statistics Canada monthly electric power statistics (tables 25-10-0001-01 to 2007, 25-10-0015-01 and
25-10-0016-01 from 2008), the Canada Energy Regulator's monthly electricity imports and exports, and daily
mean temperatures from the Government of Canada's climate data service.

Run from the repository folder: python balance.py
"""
from pathlib import Path
import calendar
import datetime as dt
import io
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.mlemodel import MLEModel

DATA, FIGURES, OUTPUT = Path("data"), Path("figures"), Path("output")
START = "1980-01"


def cached(url, name, tries=3, wait=30):
    """Download once into data/; delete the file to download it again. A failed download (the climate service
    sometimes times out) is retried after 30 and then 60 seconds."""
    path = DATA / name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        for attempt in range(1, tries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    body = response.read()  # read in full before writing, so a failure leaves no partial file
                break
            except OSError as error:  # timeouts, refused connections and HTTP errors
                if attempt == tries:
                    raise
                print(f"Download of {name} failed ({error}); retrying in {wait * attempt} seconds")
                time.sleep(wait * attempt)
        path.write_bytes(body)
    return path


def statcan(pid):
    """A full Statistics Canada table, from its CSV download."""
    with zipfile.ZipFile(cached(f"https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip", f"{pid}.zip")) as z:
        name = next(n for n in z.namelist() if not n.endswith("MetaData.csv"))
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


# Data: electricity ----
def bc(table, *columns):
    """British Columbia's rows of a table, one column per category, indexed by month."""
    t = table[table.GEO == "British Columbia"]
    for column, value in columns[:-1]:
        t = t[t[column] == value]
    wide = t.pivot(index="REF_DATE", columns=columns[-1], values="VALUE")
    return wide.set_axis(pd.PeriodIndex(wide.index.astype(str), freq="M"))  # annual tables give years as numbers


def balance():
    """BC's monthly electricity balance in GWh: generation (total and hydro), receipts and deliveries, and
    electricity available for use in the province (generation + receipts - deliveries), which stands in for
    demand. The 1950-2007 table joins the 2008 one."""
    old = bc(statcan("25100001"), "Electric power, components")
    new = bc(statcan("25100016"), "Electric power, components")
    types = bc(statcan("25100015"), ("Class of electricity producer", "Total all classes of electricity producer"),
               "Type of electricity generation")
    b = pd.concat([
        pd.DataFrame({"demand": old["Total available"], "generation": old["Overall total generation"],
                      "hydro": old["Total hydro generation"], "receipts": old["Total receipts"],
                      "deliveries": old["Total deliveries"]}),
        pd.DataFrame({"demand": new["Total electricity available for use within specific geographic border"],
                      "generation": new["Total generation"], "hydro": types["Hydraulic turbine"],
                      "receipts": new["Total receipts"], "deliveries": new["Total deliveries"]}),
    ]) / 1000
    b["other_generation"] = b.generation - b.hydro
    b["net_imports"] = b.receipts - b.deliveries
    return b.loc[START:]


def trade():
    """BC's monthly electricity trade with the United States from the Canada Energy Regulator: GWh and
    millions of dollars, imports and exports."""
    path = cached("https://www.cer-rec.gc.ca/open/imports-exports/electricity-exports-and-imports-monthly.csv",
                  "cer_trade.csv")
    t = pd.read_csv(path, encoding="latin-1")
    t = t[((t.Activity == "Exports") & (t.Source == "British Columbia") & (t.Destination != "Total"))
          | ((t.Activity == "Imports") & (t.Destination == "British Columbia") & (t.Source != "Total"))]
    t = t.assign(month=pd.PeriodIndex(pd.to_datetime(t.Period, format="%m/%d/%Y"), freq="M"))
    sums = t.groupby(["month", "Activity"])[["Energy (MW.h)", "Total Value (CAN$)"]].sum().unstack()
    return pd.DataFrame({"export_gwh": sums["Energy (MW.h)", "Exports"] / 1e3,
                         "import_gwh": sums["Energy (MW.h)", "Imports"] / 1e3,
                         "export_value": sums["Total Value (CAN$)", "Exports"] / 1e6,
                         "import_value": sums["Total Value (CAN$)", "Imports"] / 1e6})


def capacity():
    """BC's installed generating capacity at year end, MW by type (2012 is not published)."""
    cap = bc(statcan("25100022"), ("Class of electricity producer", "Total all classes of electricity producer"),
             "Type of electricity generation")
    return cap.set_axis(cap.index.year) / 1000


def fuel_generation():
    """BC's annual generation from all combustible fuels (gas, oil and biomass; published from 2016) and from
    fossil fuels alone (from 2020), TWh."""
    g = bc(statcan("25100015"), ("Class of electricity producer", "Total all classes of electricity producer"),
           "Type of electricity generation")
    year = g.groupby(g.index.year).sum(min_count=12) / 1e6
    return pd.DataFrame({"thermal_twh": year["Total electricity production from combustible fuels"],
                         "fossil_twh": year["Total electricity production from non-renewable combustible fuels"]})


def capacity_factor(twh, mw):
    """Output as a share of what the capacity would produce running all year."""
    hours = pd.Series([24 * (365 + calendar.isleap(y)) for y in twh.index], index=twh.index)
    return twh * 1e6 / (mw * hours)


# Data: hourly flows with the United States ----
# The US Energy Information Administration's Form EIA-930 files, one per half-year, give each US balancing
# authority's hourly flow with its neighbours. BC Hydro's only reported US neighbour is the Bonneville Power
# Administration; before 10 May 2017 its rows are placeholder zeros. Past half-years are kept in the repository
# (flows/), so a run downloads only the later files, about 100 MB each.
EIA = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_INTERCHANGE_{}_{}.csv"
FLOWS = Path("flows") / "bc_us_hourly.csv.gz"
FLOWS_START = "2017-05-10"
TIME = "%m/%d/%Y %I:%M:%S %p"


def eia_rows(path):
    """MW into BC each hour from one EIA-930 interchange file (negative: out of BC), indexed by UTC hour ending,
    with the Pacific hour ending. The US side reports, and a positive flow is out of the US balancing authority."""
    t = pd.read_csv(path, thousands=",", dtype={"Interchange (MW)": float}, usecols=[
        "Directly Interconnected Balancing Authority", "Interchange (MW)", "Local Time at End of Hour",
        "UTC Time at End of Hour"])
    t = t[t["Directly Interconnected Balancing Authority"] == "BCHA"]
    t = pd.DataFrame({"local": pd.to_datetime(t["Local Time at End of Hour"], format=TIME).to_numpy(),
                      "mw": t["Interchange (MW)"].to_numpy()},
                     index=pd.to_datetime(t["UTC Time at End of Hour"], format=TIME).rename("utc"))
    return pd.DataFrame({"local": t.local.groupby(level=0).first(), "mw": t.mw.groupby(level=0).sum(min_count=1)})


def half_years(since, today=None):
    """EIA's six-month files, as (year, half), from the one holding `since` to the one holding today."""
    months = pd.period_range(pd.Timestamp(since).to_period("M"), pd.Timestamp(today or dt.date.today()), freq="M")
    return sorted({(m.year, "Jan_Jun" if m.month <= 6 else "Jul_Dec") for m in months})


def flows(today=None):
    """BC's hourly flows with the United States: the stored past half-years plus every later one to today."""
    past = pd.read_csv(FLOWS, index_col="utc", parse_dates=["utc", "local"])
    new = [eia_rows(cached(EIA.format(year, half), f"eia_{year}_{half}.csv"))
           for year, half in half_years(past.index.max(), today)]
    f = pd.concat([past, *new])
    f = f[~f.index.duplicated(keep="last")].sort_index()  # a newer file replaces stored hours it repeats
    return f[f.local > FLOWS_START]  # from the first hour of 10 May 2017


# Data: temperature ----
# One airport per region; each station was replaced between 2009 and 2013, so two IDs per city (newest first)
STATIONS = {
    "Vancouver": ("1108395", "1108447"),
    "Victoria": ("1018621", "1018620"),
    "Kamloops": ("1163781", "1163780"),
    "Prince George": ("1096439", "1096450"),
}


def temperature(climate_id):
    # The service returns at most 10,000 days per request, so two requests per station
    parts = []
    for start, end in [("1979-12-01", "2003-12-31"), ("2004-01-01", "2026-12-31")]:
        url = ("https://api.weather.gc.ca/collections/climate-daily/items?f=csv&sortby=LOCAL_DATE&limit=10000"
               f"&CLIMATE_IDENTIFIER={climate_id}&datetime={start}/{end}&properties=LOCAL_DATE,MEAN_TEMPERATURE")
        path = cached(url, f"climate_{climate_id}_{start[:4]}.csv")
        if path.stat().st_size > 0:  # empty when the station was not open in that window
            parts.append(pd.read_csv(path))
    t = pd.concat(parts)
    return t.set_index(pd.to_datetime(t.LOCAL_DATE).dt.normalize()).MEAN_TEMPERATURE.dropna()


def degree_days(daily):
    """Monthly heating and cooling degree days below and above 18 C, from months with at least two weeks of
    readings, scaled up to the full month."""
    days = pd.DataFrame({"hdd": (18 - daily).clip(lower=0), "cdd": (daily - 18).clip(lower=0)})
    month = days.groupby(days.index.to_period("M"))
    full = month.mean().mul(month.size().index.days_in_month, axis=0)
    return full[month.size() >= 14]


def fill_from_others(stations):
    """A station's missing month (two at Kamloops in 1992-1993) is the other stations' average that month plus
    the station's usual gap to them in that calendar month."""
    filled = {}
    for city, d in stations.items():
        others = sum(v for c, v in stations.items() if c != city) / (len(stations) - 1)
        gap = (d - others).groupby((d - others).index.month).mean()
        estimate = others + gap.reindex(others.index.month).set_axis(others.index)
        filled[city] = d.reindex(others.index).fillna(estimate)
    return filled


def weather():
    """Heating and cooling degree days averaged over the four airports."""
    stations = {city: degree_days(temperature(new).combine_first(temperature(old)))
                for city, (new, old) in STATIONS.items()}
    return sum(fill_from_others(stations).values()) / len(stations)


# Demand and weather ----
PERIODS = [(1980, 1994), (1995, 2009), (2010, 2026)]
NORMAL = ("1991-01", "2020-12")


def monthly(b, w):
    """Average load in MW (demand spread over the month's hours) with heating and cooling degrees per day."""
    days = b.index.days_in_month
    d = pd.DataFrame({"load": b.demand * 1000 / (24 * days),
                      "hdd": w.hdd.reindex(b.index) / days, "cdd": w.cdd.reindex(b.index) / days})
    normal = d.loc[NORMAL[0]:NORMAL[1]].groupby(d.loc[NORMAL[0]:NORMAL[1]].index.month)[["hdd", "cdd"]].mean()
    return d.join(normal.reindex(d.index.month).set_axis(d.index).add_suffix("_normal"))


def weather_columns(d):
    """Heating and cooling degrees per day, with a separate effect for each period."""
    year = d.index.year
    return pd.concat({f"{kind}_{a}": d[kind].where((year >= a) & (year <= z), 0)
                      for a, z in PERIODS for kind in ("hdd", "cdd")}, axis=1)


def demand_model(d):
    """Least squares of average load on a level for each year, calendar month and the period's heating and
    cooling effects; Newey-West standard errors (12 lags)."""
    d = d.dropna(subset=["load", "hdd", "cdd"])
    X = pd.concat([pd.get_dummies(d.index.year, prefix="year", dtype=float).set_axis(d.index),
                   pd.get_dummies(d.index.month, prefix="month", drop_first=True, dtype=float).set_axis(d.index),
                   weather_columns(d)], axis=1)
    return sm.OLS(d.load, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def normalised(d, fit):
    """Average load as if each month had its 1991-2020 normal weather, using the period's effects."""
    actual = weather_columns(d) @ fit.params[weather_columns(d).columns]
    normal = weather_columns(d.assign(hdd=d.hdd_normal, cdd=d.cdd_normal)) @ fit.params[weather_columns(d).columns]
    return d.load - actual + normal


def gwh(mw):
    """Monthly average MW to GWh in the month."""
    return mw * 24 * mw.index.days_in_month / 1000


def annual_twh(mw):
    """Monthly average MW to annual TWh, full years only."""
    twh = mw * 24 * mw.index.days_in_month / 1e6
    year = twh.groupby(twh.index.year)
    return year.sum()[year.count() == 12]


def full_years(frame):
    """Sums over the calendar years in which every column has all 12 months."""
    year = frame.groupby(frame.index.year)
    return year.sum()[year.count().min(axis=1) == 12]


# The drought ----
BASELINE = (2013, 2022)  # the ten years before the drought


def yearly(b, normal_mw):
    """The balance in TWh for each full year, with weather-normalised demand and the weather's part."""
    year = (full_years(b) / 1000).assign(normalised=annual_twh(normal_mw)).rename_axis("year")
    return year.assign(weather=year.demand - year.normalised)


def drought(b, normal_mw):
    """Each year's change from the baseline average in net imports (receipts minus deliveries), and its parts:
    net imports = demand - hydro - other generation, and demand = weather-normalised demand + weather."""
    year = yearly(b, normal_mw)
    change = year - year.loc[BASELINE[0]:BASELINE[1]].mean()
    return pd.DataFrame({"net_imports": change.net_imports, "demand_growth": change.normalised,
                         "weather": change.weather, "less_hydro": -change.hydro,
                         "less_other_generation": -change.other_generation})


def trade_by_year(t):
    """Annual BC-US trade: TWh, millions of dollars and average price ($ per MWh) each way."""
    y = full_years(t)
    return y.assign(export_twh=y.export_gwh / 1000, import_twh=y.import_gwh / 1000,
                    export_price=y.export_value / y.export_gwh * 1000, import_price=y.import_value / y.import_gwh * 1000,
                    net_value=y.export_value - y.import_value)


# Time-varying temperature response ----
class TimeVaryingLoad(MLEModel):
    """Monthly average load = level + heating effect x heating degrees + cooling effect x cooling degrees
    + calendar-month effects + an AR(1) error. The level is a smooth trend and the two weather effects are
    random walks that can move each January, so the Kalman smoother traces how they changed from year to year.
    The shock variances and the error's persistence are estimated by maximum likelihood."""

    def __init__(self, load, heating, cooling, calendar_X):
        super().__init__(np.asarray(load, float), k_states=5 + calendar_X.shape[1], k_posdef=4)
        # States: level, slope, heating effect, cooling effect, AR(1) error, then the calendar effects (constant)
        design = np.zeros((1, self.k_states, self.nobs))
        design[0, 0] = design[0, 4] = 1
        design[0, 2], design[0, 3] = np.asarray(heating, float), np.asarray(cooling, float)
        design[0, 5:] = np.asarray(calendar_X, float).T
        self["design"] = design
        self["transition"] = np.eye(self.k_states)
        self["transition", 0, 1] = 1  # the level moves by the slope each month
        # Shocks: to the slope and the error every month, to the weather effects each January
        selection = np.zeros((self.k_states, 4, self.nobs))
        selection[1, 0] = selection[4, 3] = 1
        selection[2, 1] = selection[3, 2] = pd.PeriodIndex(load.index).month == 1
        self["selection"] = selection
        self.ssm.initialize_approximate_diffuse(1e8)
        self.loglikelihood_burn = self.k_states

    @property
    def param_names(self):
        return ["slope", "heating", "cooling", "error", "persistence"]

    @property
    def start_params(self):
        return np.array([1.0, 100.0, 100.0, 0.5 * np.var(np.diff(self.endog[:, 0])), 0.5])

    def transform_params(self, unconstrained):
        return np.r_[unconstrained[:4] ** 2, np.tanh(unconstrained[4])]

    def untransform_params(self, constrained):
        return np.r_[constrained[:4] ** 0.5, np.arctanh(constrained[4])]

    def update(self, params, **kwargs):
        slope, heating, cooling, error, persistence = super().update(params, **kwargs)
        self["state_cov"] = np.diag([slope, heating, cooling, error])
        self["transition", 4, 4] = persistence


def temperature_effect(d):
    """Each year's heating and cooling effect (MW per degree at all four airports) with 90% bands."""
    calendar_X = pd.get_dummies(d.index.month, prefix="month", drop_first=True, dtype=float)
    model = TimeVaryingLoad(d.load, d.hdd, d.cdd, calendar_X)
    fit = model.fit(disp=False, maxiter=2000)
    january = d.index.month == 1
    out = {}
    for name, i in [("heating", 2), ("cooling", 3)]:
        effect, sd = fit.smoothed_state[i], fit.smoothed_state_cov[i, i] ** 0.5
        out[name] = pd.DataFrame({"effect": effect, "low": effect - 1.645 * sd, "high": effect + 1.645 * sd},
                                 index=d.index.year)[january]
    return pd.concat(out, axis=1), fit


# Hourly flows ----
FLOWS_BASELINE = (2018, 2022)  # the full years before the drought
EVENING, OVERNIGHT = range(17, 21), range(0, 6)  # hours beginning 5-9 pm and midnight-6 am, Pacific time


def hour_start(f):
    """The Pacific time each hour begins (EIA labels hours by their end)."""
    return f.local - pd.Timedelta(hours=1)


def flows_by_year(f):
    """Each year's net flow into BC (TWh), share of hours importing, and the flow in the busiest 1% of hours
    each way (MW)."""
    year = hour_start(f).dt.year
    g = f.mw.groupby(year)
    return pd.DataFrame({"hours": g.count(), "net_into_bc_twh": g.sum() / 1e6,
                         "share_hours_importing": (f.mw > 0).groupby(year).mean(),
                         "import_top_1pct_mw": g.quantile(0.99), "export_top_1pct_mw": -g.quantile(0.01)}
                        ).rename_axis("year")


def flows_by_hour(f):
    """Average MW into BC by Pacific hour beginning, for the baseline years together and each later year."""
    start = hour_start(f)
    year = start.dt.year
    keep = year >= FLOWS_BASELINE[0]
    group = year.astype(str).where(year > FLOWS_BASELINE[1], f"{FLOWS_BASELINE[0]}-{FLOWS_BASELINE[1]}")
    return (f.mw[keep].groupby([start.dt.hour[keep], group[keep]]).mean().unstack()
            .rename_axis(index="hour", columns=None))


# Charts, in a style shared with rivers.py, trading.py, forecast.py and dashboard.py ----
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#a8a7a2"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#3a9b6a", "#c8413b"
STYLE = {
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 12,
}
SOURCE = ("Source: Statistics Canada tables 25-10-0001-01, 25-10-0015-01 and 25-10-0016-01; "
          "Government of Canada daily climate data.")


def save(fig, name, title, subtitle, caption, folder=FIGURES, dpi=200):
    fig.text(0.01, 0.99, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, subtitle, color=MUTED, va="top")
    fig.text(0.01, 0.01, caption, color=MUTED, fontsize=9, va="bottom")
    fig.savefig(folder / name, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    for folder in (FIGURES, OUTPUT):
        folder.mkdir(exist_ok=True)

    b = balance()
    d = monthly(b, weather())
    fit = demand_model(d)
    normal_mw = normalised(d, fit)
    annual = yearly(b, normal_mw)
    cap = capacity()
    annual["hydro_capacity_mw"] = cap["Hydraulic turbine"]
    annual["hydro_capacity_factor"] = capacity_factor(annual.hydro, annual.hydro_capacity_mw)
    annual = annual.join(fuel_generation())
    annual["thermal_capacity_factor"] = capacity_factor(annual.thermal_twh, cap["Total thermal"])
    parts = drought(b, normal_mw).rename_axis("year")
    monthly_trade = trade()
    t = trade_by_year(monthly_trade).rename_axis("year")
    f = flows()
    by_year, by_hour = flows_by_year(f), flows_by_hour(f)
    eia_month = f.mw.groupby(hour_start(f).dt.to_period("M")).sum() / 1000
    cer_month = monthly_trade.import_gwh - monthly_trade.export_gwh
    check = pd.concat({"eia": eia_month, "cer": cer_month}, axis=1).dropna()
    effect, kalman = temperature_effect(d)
    constant = kalman.model.fit_constrained({"heating": 0.0}, disp=False, maxiter=2000)
    lr = max(2 * (kalman.llf - constant.llf), 0)
    p_heating = 0.5 * stats.chi2.sf(lr, 1)  # the variance is on the boundary under the null

    annual.round(2).to_csv(OUTPUT / "annual.csv")
    parts.round(2).to_csv(OUTPUT / "drought.csv")
    t.round(2).to_csv(OUTPUT / "trade.csv")
    by_year.round(3).to_csv(OUTPUT / "flows_by_year.csv")
    by_hour.round(1).to_csv(OUTPUT / "flows_by_hour.csv")
    effect.set_axis([f"{a}_{b}" for a, b in effect.columns], axis=1).rename_axis("year").round(1).to_csv(
        OUTPUT / "temperature_effect.csv")
    periods = fit.params.filter(regex="^(hdd|cdd)_")
    print(f"Demand model R2 {fit.rsquared:.3f}; heating and cooling MW per degree by period:")
    print(pd.DataFrame({"effect": periods, "se": fit.bse[periods.index]}).round(0).to_string())
    print(annual[["demand", "normalised", "weather", "hydro", "net_imports"]].round(1).to_string())
    print(parts.loc[2019:].round(1).to_string())
    print(t.loc[2019:, ["export_twh", "import_twh", "export_value", "import_value", "export_price",
                        "import_price", "net_value"]].round(1).to_string())
    print(effect.loc[[1980, 1990, 2000, 2010, 2020, effect.index.max()]].round(0).to_string())
    print(f"Heating effect constant over time? LR {lr:.1f}, p = {p_heating:.2g}")
    print(annual[["hydro_capacity_mw", "hydro_capacity_factor", "thermal_capacity_factor", "fossil_twh"]]
          .loc[2016:].round(2).to_string())
    print(by_year.round(2).to_string())
    print(by_hour.round(0).to_string())
    print(f"EIA hourly flows against CER monthly net trade, {check.index.min()} to {check.index.max()}: "
          f"correlation {check.corr().iloc[0, 1]:.3f}, mean absolute difference {(check.eia - check.cer).abs().mean():.0f} GWh")

    # Figures ----
    plt.rcParams.update(STYLE)

    # Annual demand, actual and weather-normalised
    first, last = annual.index.min(), annual.index.max()
    growth = annual.normalised[last] / annual.normalised[first] - 1
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(top=0.80, bottom=0.17)
    ax.plot(annual.index, annual.demand, color=GREY, lw=1.6, label="Actual")
    ax.plot(annual.index, annual.normalised, color=BLUE, lw=2.4, label="Weather-normalised")
    ax.annotate("2015: low in the source\ndata, see Limits", (2015, annual.demand[2015]), (2008.5, 53),
                color=MUTED, fontsize=9, arrowprops={"arrowstyle": "-", "color": GREY})
    ax.set_ylabel("TWh a year")
    ax.legend(loc="upper left")
    save(fig, "annual_demand.png",
         f"BC's electricity demand has grown {growth:.0%} since {first}; weather moves a year by under "
         f"{np.ceil(annual.weather.abs().max()):.0f} TWh",
         "Electricity available for use in British Columbia, actual and with each month's 1991-2020 normal weather",
         "Weather normalisation: monthly model of average load on heating and cooling degree days at four airports, "
         "calendar month and a level for each year.\n" + SOURCE)

    # The drought: generation, demand and the breakdown of net imports
    share = annual.net_imports[2023] / annual.demand[2023]
    hydro_drop = parts.less_hydro[2023] / annual.hydro.loc[BASELINE[0]:BASELINE[1]].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.27, wspace=0.25)
    y = annual.loc[1990:]
    left.fill_between(y.index, y.generation, y.demand, where=y.demand > y.generation, interpolate=True,
                      color=RED, alpha=0.25, lw=0, label="Net imports")
    left.fill_between(y.index, y.generation, y.demand, where=y.demand <= y.generation, interpolate=True,
                      color=GREEN, alpha=0.2, lw=0, label="Net exports")
    left.plot(y.index, y.demand, color=INK, lw=2.2, label="Demand")
    left.plot(y.index, y.generation, color=BLUE, lw=2.2, label="Generation in BC")
    left.plot(y.index, y.hydro, color=BLUE, lw=1.4, ls="--", label="of which hydro")
    left.set_ylabel("TWh a year")
    left.set_ylim(40, None)
    left.legend(loc="lower left", ncol=3, fontsize=10)
    left.set_title("Generation, demand and trade")
    years = [yr for yr in (2023, 2024, 2025) if yr in parts.index]
    stack = [("demand_growth", "Demand growth", ORANGE), ("weather", "Weather", GREY),
             ("less_hydro", "Lower hydro output", BLUE), ("less_other_generation", "Other generation", GREEN)]
    for i, yr in enumerate(years):
        up = down = 0.0
        for key, label, colour in stack:
            v = parts.loc[yr, key]
            right.bar(i, v, bottom=up if v >= 0 else down, color=colour, width=0.6, label=label if i == 0 else None)
            up, down = (up + v, down) if v >= 0 else (up, down + v)
        right.plot(i, parts.loc[yr, "net_imports"], "o", color=INK, ms=7, label="Net imports" if i == 0 else None)
    right.axhline(0, color=MUTED, lw=0.8)
    right.set_xticks(range(len(years)), years)
    right.set_ylabel(f"TWh, change from {BASELINE[0]}-{BASELINE[1]} average")
    right.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=9)
    right.set_title("Why net imports rose")
    save(fig, "drought.png",
         f"Low water cut BC's hydro output {hydro_drop:.0%} in 2023, and it imported {share:.0%} of the power it used",
         f"Net imports are demand minus generation in BC. Right: each year's change from the {BASELINE[0]}-{BASELINE[1]} "
         "average, split into its parts",
         "Demand growth is the change in weather-normalised demand. Trade includes other provinces and the United States.\n"
         + SOURCE)

    # Trade with the United States: value and prices
    y = t.loc[1990:]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.22)
    left.plot(y.index, y.export_value, color=GREEN, lw=2.2, label="Exports")
    left.plot(y.index, y.import_value, color=RED, lw=2.2, label="Imports")
    left.set_ylabel("$ million a year")
    left.legend(loc="upper left")
    left.set_title("Value of trade with the US")
    right.plot(y.index, y.export_price, color=GREEN, lw=2.2, label="Price received for exports")
    right.plot(y.index, y.import_price, color=RED, lw=2.2, label="Price paid for imports")
    right.set_ylabel("$ per MWh")
    right.legend(loc="upper left")
    right.set_title("Average prices")
    save(fig, "trade.png",
         f"In 2023 BC paid ${t.import_value[2023] / 1000:.1f} billion for US power, but still sold its own exports "
         "for more per MWh",
         "British Columbia's electricity trade with the United States, current dollars",
         "Source: Canada Energy Regulator, Electricity Imports and Exports (monthly). "
         "Contains information licensed under the Open Government Licence - Canada.")

    # Temperature response over time
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.22)
    for ax, (name, key, kind, colour) in zip(axes, [("Heating: a degree colder", "heating", "hdd", BLUE),
                                                     ("Cooling: a degree hotter", "cooling", "cdd", ORANGE)]):
        e = effect[key]
        ax.fill_between(e.index, e.low, e.high, color=colour, alpha=0.18, lw=0, label="90% band")
        ax.plot(e.index, e.effect, color=colour, lw=2.2, label="Changing year by year")
        for a, z in PERIODS:
            z = min(z, e.index.max())
            ax.plot([a, z], [periods[f"{kind}_{a}"]] * 2, color=MUTED, lw=1.4, ls="--",
                    label="Fixed for each period" if a == PERIODS[0][0] else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(name)
    axes[0].set_ylabel("MW of average load per degree")
    axes[0].legend(loc="upper left", fontsize=10)
    h = effect["heating", "effect"]
    p_text = "< 0.001" if p_heating < 0.001 else f"= {p_heating:.3f}"
    save(fig, "temperature_effect.png",
         f"A degree colder now adds about {h[last]:-.0f} MW to BC's demand, around {h[last] / h[first]:.0f} times as "
         f"much as in {first}",
         "Average load added by a month that is one degree colder (or hotter) every day at all four airports, "
         "estimated with a Kalman filter",
         f"Test of change over time (heating): p {p_text}. Airports: Vancouver, Victoria, Kamloops, Prince George.\n"
         + SOURCE)

    # How hard the hydro fleet ran, and installed capacity
    cf = annual.hydro_capacity_factor.dropna() * 100
    low = cf.idxmin()
    base = cf.loc[BASELINE[0]:BASELINE[1]].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.22)
    left.bar(cf.index, cf, color=[RED if y == low else BLUE for y in cf.index], width=0.7)
    left.axhline(base, color=MUTED, ls="--", lw=1.2, label=f"{BASELINE[0]}-{BASELINE[1]} average, {base:.0f}%")
    left.set_xticks(range(cf.index.min(), cf.index.max() + 1, 4))
    left.set_ylabel("Hydro output, % of capacity")
    left.set_ylim(0, cf.max() * 1.15)
    left.legend(loc="upper right", fontsize=10)
    left.set_title("How hard the hydro fleet ran")
    mix = cap.loc[cap["Total installed capacity"].notna(),
                  ["Hydraulic turbine", "Total thermal", "Wind power turbine", "Solar power"]].fillna(0) / 1000
    right.stackplot(mix.index, mix.T, colors=[BLUE, GREY, GREEN, ORANGE], labels=["Hydro", "Thermal", "Wind", "Solar"])
    right.set_ylim(0, mix.sum(axis=1).max() * 1.2)
    right.set_ylabel("GW at year end")
    right.legend(loc="upper left", fontsize=10, ncol=4)
    right.set_title("Installed capacity")
    save(fig, "hydro_capacity.png",
         f"In {low} BC's hydro fleet produced {cf[low]:.0f}% of what it could, the least of any year since "
         f"{cf.index.min()}",
         "Hydro generation as a share of what the installed hydro capacity would produce running all year",
         "Site C does not yet appear in the 2024-2025 capacity. 2012 not published.\n"
         "Source: Statistics Canada tables 25-10-0001-01, 25-10-0015-01 and 25-10-0022-01.")

    # Hourly flows with the United States
    COLOURS = [RED, ORANGE, "#7a5bb5", GREEN, BLUE]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.27, wspace=0.22)
    left.axvspan(EVENING[0] - 0.5, EVENING[-1] + 0.5, color=GREY, alpha=0.15, lw=0)
    for i, column in enumerate(by_hour):
        baseline = i == 0
        label = column if by_year.hours.get(int(column[-4:]), 0) >= 8760 or baseline else f"{column} so far"
        left.plot(by_hour.index, by_hour[column], color=GREY if baseline else COLOURS[(i - 1) % len(COLOURS)],
                  lw=2.8 if baseline else 2.2, label=label)
    left.axhline(0, color=MUTED, lw=0.8)
    left.set_xticks([0, 6, 12, 18, 23], ["midnight", "6 am", "noon", "6 pm", "11 pm"])
    left.set_ylabel("MW into BC (negative: out of BC)")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=5)
    left.set_title("Average flow by hour of the day (5-9 pm shaded)")
    share = by_year.share_hours_importing.loc[FLOWS_BASELINE[0]:] * 100
    right.bar(share.index, share, color=[RED if s > 50 else BLUE for s in share], width=0.7)
    right.axhline(50, color=MUTED, lw=0.8)
    right.set_ylabel("% of hours")
    right.set_title("Hours BC was importing")
    save(fig, "hourly_flows.png",
         "Even in the drought, BC imported least at the morning and evening peaks",
         "Hourly flow between the Bonneville Power Administration and BC Hydro, by Pacific hour",
         f"{f.local.max().year}: to {hour_start(f).max():%d %B}. Monthly totals match the Canada Energy Regulator's net trade "
         f"(correlation {check.corr().iloc[0, 1]:.3f}).\nSource: US Energy Information Administration, Form EIA-930, "
         f"{dt.date.today():%B %Y}.")


if __name__ == "__main__":
    main()
