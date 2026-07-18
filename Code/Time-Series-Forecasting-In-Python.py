# Title: Time Series Forecasting in Python
# Author: Alexander Zakrzeski
# Date: July 16, 2026

import polars as pl

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL

# Part 1. Time Waits for No One

# 1.1 Understanding Time Series Forecasting

jj = (
    pl.read_parquet("Johnson-And-Johnson-Quarterly-EPS.parquet")
      .rename({"data": "eps"}) 
      .with_columns(pl.col("date").str.to_date())
      .with_columns(pl.col("date").dt.year().alias("year"))  
      .select("date", "year", "eps")
    )

fig, ax = plt.subplots()
ax.plot(jj["date"], jj["eps"])
ax.set_title("Johnson & Johnson Quarterly Earnings per Share")
ax.set_xlabel("Date")
ax.set_ylabel("Earnings per share (USD)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

stl_decomposition = STL(jj["eps"], period = 4).fit()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols = 1, sharex = True)
ax1.plot(jj["date"], stl_decomposition.observed)
ax1.set_ylabel("Observed")
ax1.set_title("Johnson & Johnson EPS — STL Decomposition")
ax2.plot(jj["date"], stl_decomposition.trend)
ax2.set_ylabel("Trend")
ax3.plot(jj["date"], stl_decomposition.seasonal)
ax3.set_ylabel("Seasonal")
ax4.plot(jj["date"], stl_decomposition.resid)
ax4.set_ylabel("Residuals")
ax4.xaxis.set_major_locator(mdates.YearLocator(2))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# 1.2 A Naive Prediction of the Future

os.chdir("/Users/azak13/Desktop/Time-Series-In-Python/Data")