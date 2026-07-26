# Title: Time Series Forecasting in Python
# Author: Alexander Zakrzeski
# Date: July 26, 2026

import polars as pl

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL

from sklearn.metrics import mean_absolute_percentage_error

# Part 1. Time Waits for No One

# 1.1 Understanding Time Series Forecasting

# Create a DataFrame containing the time series
jj = (
    pl.read_parquet("Johnson-And-Johnson-Quarterly-EPS.parquet")
      .rename({"data": "eps"}) 
      .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
      .with_columns(pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.quarter().alias("quarter")) 
      .select("date", "year", "quarter", "eps")
    )

# Create a plot visualizing the time series
fig, ax = plt.subplots()
ax.plot(jj["date"], jj["eps"])
ax.set_title("Johnson & Johnson Quarterly Earnings per Share")
ax.set_xlabel("Date")
ax.set_ylabel("Earnings per share (USD)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout()
plt.show()

# Decompose the time series into trend, seasonal, and residual components
stl_decomposition = STL(jj["eps"], period = 4).fit()

# Create a plot visualizing the time series decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, sharex = True)
ax1.plot(jj["date"], stl_decomposition.observed)
ax1.set_title("STL Decomposition of Johnson & Johnson EPS")
ax1.set_ylabel("Observed")
ax2.plot(jj["date"], stl_decomposition.trend)
ax2.set_ylabel("Trend")
ax3.plot(jj["date"], stl_decomposition.seasonal)
ax3.set_ylabel("Seasonal")
ax4.plot(jj["date"], stl_decomposition.resid)
ax4.set_ylabel("Residuals")
ax4.xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout()
plt.show()

# 1.2 A Naive Prediction of the Future

# Create the training and test sets and generate the naive seasonal forecast
jj_train1 = jj.filter(pl.col("year") <= 1979)

jj_test1 = (
    jj.filter(pl.col("year") == 1980)
      .join(jj_train1.filter(pl.col("year") == 1979)
                     .rename({"eps": "prediction"})
                     .select("quarter", "prediction"), 
            on = "quarter", how = "left")
    )

# Calculate the mean absolute percentage error
print(f"{mean_absolute_percentage_error(jj_test1["eps"], 
                                        jj_test1["prediction"]):.2%}")

# Create a plot comparing the actual and forecasted values
fig, ax = plt.subplots()
ax.plot(jj_train1["date"], jj_train1["eps"], "g-.", label = "Train")
ax.plot(jj_test1["date"], jj_test1["eps"], "b-", label = "Test")
ax.plot(jj_test1["date"], jj_test1["prediction"], "r--", label = "Predicted")
ax.set_title("Naive Seasonal Forecast of Johnson & Johnson EPS")
ax.set_xlabel("Date")
ax.set_ylabel("Earnings per share (USD)")
ax.axvspan(jj_test1["date"].min(), jj_test1["date"].max(), color = "#808080", 
           alpha = 0.2)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend(loc = "upper left")
plt.tight_layout()
plt.show()

# 1.3 Going on a Random Walk


os.chdir("/Users/azak13/Desktop/Time-Series-In-Python/Data")