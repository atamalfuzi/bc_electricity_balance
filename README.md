# bc_electricity_balance

[![tests](https://github.com/atamalfuzi/bc_electricity_balance/actions/workflows/tests.yml/badge.svg)](https://github.com/atamalfuzi/bc_electricity_balance/actions/workflows/tests.yml)

An independent look at British Columbia's electricity balance since 1980, built from open data in Python. It asks four questions: why BC imported so much power in 2023–2025, what that trade cost, how much of the growth in demand is weather, and how demand's response to temperature has changed.

**[Live dashboard](https://atamalfuzi.github.io/bc_electricity_balance/)**, rebuilt on the 15th of each month from the latest Statistics Canada and Canada Energy Regulator data: the latest month's demand against what the weather and season predict, hydro output and net imports over the last 12 months, and what the trade cost. The results below are as of October 2026 (data through July 2026).

> **Personal project.** This is my own analysis of publicly available data. It is not associated with my work for any current or past employer and does not represent their views.

![Generation, demand and the 2023-2025 import swing](figures/drought.png)

![Value and prices of BC's electricity trade with the United States](figures/trade.png)

![Annual demand, actual and weather-normalised](figures/annual_demand.png)

![Temperature effect over time, Kalman filter](figures/temperature_effect.png)

## What it shows

- **In 2023 low water cut BC's hydro output by 19%, and BC imported 17% of the power it used.** Hydro generation fell to 49.1 TWh, against 60.4 TWh a year in 2013–2022, and net imports (receipts minus deliveries, from other provinces and the United States) reached 11.2 TWh, against average net exports of 3.1 TWh a year in 2013–2022. Of the 14.3 TWh swing, lower hydro output accounts for 11.3 TWh and growth in weather-normalised demand for 3.5 TWh; the mild weather took off 0.7 TWh.
- **By 2025 demand growth, not water, kept imports high.** Net imports were still 8.7 TWh above the 2013–2022 average in 2025, and 6.0 TWh of that is demand growth, against 3.1 TWh of lower hydro output (which includes the new Site C dam). In the 12 months to July 2026, with hydro output 9% above that average, BC was back to net exports of 2.7 TWh.
- **BC still sells high and buys low, but the drought turned trade into a cost.** In 2023 BC paid $1.52 billion for 16.6 TWh of US power ($92 per MWh) and earned $1.11 billion for 6.6 TWh of exports ($168 per MWh): a $411 million deficit, against a $972 million surplus in 2022. Trade broke even in 2024 and was back to a $196 million surplus in 2025.
- **Demand has grown two thirds since 1980, and weather explains little of it.** Weather-normalised demand rose from 42.3 TWh in 1980 to 69.1 TWh in 2007, fell to about 61 TWh by 2014 and rose again to 70.2 TWh in 2025. Weather moved any single year by at most 1.2 TWh.
- **Demand responds three times as much to a cold month as in 1980.** A month that is one degree colder every day at all four airports now adds about 143 MW of average load (90% band 129–158), against 44 MW in 1980 (25–62), while demand itself grew 1.7 times. A degree hotter adds about 455 MW (355–556) in 2025, up from about 210 MW around 2000; before the mid-1990s, when hot months were rare, the cooling estimate is imprecise, and even negative in the early 1980s. Both changes are statistically clear (likelihood-ratio tests, p < 0.001). The data cannot show the cause; electric heating, heat pumps and air conditioning are the obvious candidates.

## Method

**Data preparation.** Monthly generation (total and hydro), receipts, deliveries and electricity available for use in BC come from Statistics Canada's monthly electric power tables: one for 1950–2007 and two from 2008 that report the same items. Electricity available for use (generation plus receipts minus deliveries) stands in for demand; it includes transmission losses and power that industry generates for itself. Net imports are receipts minus deliveries. The value and volume of trade with the United States come from the Canada Energy Regulator, whose volumes match Statistics Canada's. Heating and cooling degree days below and above 18 °C come from daily mean temperatures at the Vancouver, Victoria, Kamloops and Prince George airports, joining each airport's old and new station records and averaging the four. Months with fewer than 14 days of readings are dropped and the rest scaled up to the full month; the two months this drops at Kamloops (1992–1993) are filled from the other airports plus Kamloops's usual gap to them in that calendar month.

**1. Weather-normalised demand.** A least-squares regression of monthly average load (MW) on a level for each year, calendar month, and heating and cooling degrees per day with separate effects for 1980–1994, 1995–2009 and 2010 onward, with standard errors robust to a year of autocorrelation. It explains 97% of the monthly variation. Each month's weather-normalised load replaces its actual degree days with that calendar month's 1991–2020 average, at its period's effects.

**2. The import swing.** Net imports equal demand minus generation, so each year's change from the 2013–2022 average splits exactly into the change in weather-normalised demand, the change in the weather's effect, lower hydro output and lower other generation. This is an accounting split, not a statement of cause: BC Hydro chooses how much to generate and how much to buy, so lower hydro output reflects market prices and reservoir management as well as low water.

**3. Change in the temperature response.** A state-space model of monthly average load:

- a smooth trend for the structural level
- a heating effect and a cooling effect that follow random walks and can move each January
- calendar-month effects, held fixed
- an error that carries over from month to month (AR(1))

The shock variances and the error's persistence are estimated by maximum likelihood, and the Kalman smoother traces each effect with a 90% band. Likelihood-ratio tests compare it with constant effects; because the variance being tested sits on the boundary (zero), the p-values are half the chi-squared ones. The chart compares the yearly effects with the period effects from model 1.

All estimates are in [`output/`](output): annual demand, generation and trade in [`annual.csv`](output/annual.csv), the import swing in [`drought.csv`](output/drought.csv), trade in [`trade.csv`](output/trade.csv) and the yearly temperature effects in [`temperature_effect.csv`](output/temperature_effect.csv). The data handling and models are checked by the tests in [`tests/`](tests/test_balance.py), which run on every push, including the state-space model recovering a known drifting effect from simulated data.

## Limits

- **2015 looks wrong in the source data.** Statistics Canada's figure for electricity available for use in 2015 is about 4 TWh below the years either side, mostly from April to September, when reported deliveries out of BC were high. Weather does not explain it, and I have not found the cause, so 2015 is kept as reported and flagged on the chart. It sits inside the 2013–2022 baseline, where it lowers average demand by about 0.4 TWh.
- **Monthly, not hourly.** There is no openly licensed hourly demand data for BC, so the analysis cannot speak to peaks or to when in the day power flows each way, which likely explains much of the gap between export and import prices.
- **Demand is a residual.** Electricity available for use includes losses and industrial self-generation, and it moves with industry: the fall from 2007 to 2014 and the later rise include the 2009 recession and changes in industrial demand, which the year levels and the trend do not separate.
- **Four airports stand in for the province,** equally weighted, with no population weighting.
- **The cooling effect rests on few hot months,** so it is much less precise than the heating effect, and recent hot summers (2021, 2024, 2025) carry much of its rise.
- **Trade values are in current dollars** and cover trade with the United States only; trade with Alberta appears in the volumes but not the values.

## Data

The scripts download everything themselves:

| Data | Source | Licence |
|---|---|---|
| Monthly electric power statistics, 1950–2007 | Statistics Canada, [table 25-10-0001-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2510000101) | [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/terms-conditions/open-licence) |
| Monthly generation by type of electricity, 2008 onward | Statistics Canada, [table 25-10-0015-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2510001501) | Statistics Canada Open Licence |
| Monthly receipts, deliveries and availability of electricity, 2008 onward | Statistics Canada, [table 25-10-0016-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2510001601) | Statistics Canada Open Licence |
| Monthly electricity exports and imports by province and US state, volume and value, 1990 onward | Canada Energy Regulator, [Electricity – Imports and Exports](https://open.canada.ca/data/en/dataset/5c358f51-bc8c-4565-854d-9d2e35e6b178) | [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada) |
| Daily mean temperature at four airports | Government of Canada historical climate data, via the [MSC GeoMet API](https://api.weather.gc.ca/collections/climate-daily) | Open Government Licence – Canada |

Adapted from Statistics Canada, tables 25-10-0001-01, 25-10-0015-01 and 25-10-0016-01. This does not constitute an endorsement by Statistics Canada of this product. Contains information licensed under the Open Government Licence – Canada.

## Run it

Python 3.12:

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python balance.py
```

Run it from the repository folder. The first run downloads about 12 MB into `data/` and takes about a minute; later runs reuse those files (delete them to download again) and take a few seconds. It writes the charts to `figures/` and the tables to `output/`, and prints the estimates.

To run the tests: `pip install -r requirements-dev.txt`, then `pytest`. They need no downloads.

To build the dashboard, run `python dashboard.py` after `balance.py`. It writes a static site to `site/` and stops without writing if the latest data have gaps or implausible values. Delete `data/` first to pick up the newest month. The [dashboard workflow](.github/workflows/dashboard.yml) does all of this on the 15th of each month and publishes `site/` with GitHub Pages.

## License

Code: [MIT](LICENSE).
