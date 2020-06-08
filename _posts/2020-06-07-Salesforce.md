---
layout: post
title: Time-series data stationarization & data transfromations
---
***

<a href='https://en.wikipedia.org/wiki/Salesforce'> Salesforce.com, inc.</a> is an American cloud-based software company headquartered in San Francisco, California. It provides customer relationship management (CRM) service and also sells a complementary suite of enterprise applications focused on customer service, marketing automation, analytics, and application development (Wikipedia).

The data set is from <a href= 'https://www.alphavantage.co/#about'> alphavantage </a>. It is very easy to obtain the stock price information from the website. Just follow the documentation.

Before everything esle, we need data. So let's get it!


```python
# API Key:LX71H2XX0HSO5PAO 
```

Let's upload required library...


```python
import requests
import json
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import statsmodels.api as sm
import warnings
from itertools import product

warnings.filterwarnings('ignore')
# plt.style.use('seaborn-poster')
```


```python
# getting the data from aplhavantage using API key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=CRM&outputsize=full&apikey=LX71H2XX0HSO5PAO'
resp = requests.get(url)
resp.json()['Meta Data']
```




    {'1. Information': 'Daily Time Series with Splits and Dividend Events',
     '2. Symbol': 'CRM',
     '3. Last Refreshed': '2020-02-21',
     '4. Output Size': 'Full size',
     '5. Time Zone': 'US/Eastern'}



As you can see, the Meta Data node contains the data that we need.


```python
df = pd.DataFrame(resp.json()['Time Series (Daily)']) # converting json to dataframe
```


```python
df.tail() # initial look of df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2020-02-21</th>
      <th>2020-02-20</th>
      <th>2020-02-19</th>
      <th>2020-02-18</th>
      <th>2020-02-14</th>
      <th>2020-02-13</th>
      <th>2020-02-12</th>
      <th>2020-02-11</th>
      <th>2020-02-10</th>
      <th>2020-02-07</th>
      <th>...</th>
      <th>2004-07-07</th>
      <th>2004-07-06</th>
      <th>2004-07-02</th>
      <th>2004-07-01</th>
      <th>2004-06-30</th>
      <th>2004-06-29</th>
      <th>2004-06-28</th>
      <th>2004-06-25</th>
      <th>2004-06-24</th>
      <th>2004-06-23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4. close</td>
      <td>189.5000</td>
      <td>193.3600</td>
      <td>192.8700</td>
      <td>191.0900</td>
      <td>189.9500</td>
      <td>188.6400</td>
      <td>189.4600</td>
      <td>189.1100</td>
      <td>189.1200</td>
      <td>185.7200</td>
      <td>...</td>
      <td>16.3100</td>
      <td>17.0000</td>
      <td>16.9800</td>
      <td>16.0300</td>
      <td>16.0700</td>
      <td>16.4000</td>
      <td>16.0000</td>
      <td>15.8000</td>
      <td>16.7600</td>
      <td>17.2000</td>
    </tr>
    <tr>
      <td>5. adjusted close</td>
      <td>189.5000</td>
      <td>193.3600</td>
      <td>192.8700</td>
      <td>191.0900</td>
      <td>189.9500</td>
      <td>188.6400</td>
      <td>189.4600</td>
      <td>189.1100</td>
      <td>189.1200</td>
      <td>185.7200</td>
      <td>...</td>
      <td>4.0775</td>
      <td>4.2500</td>
      <td>4.2450</td>
      <td>4.0075</td>
      <td>4.0175</td>
      <td>4.1000</td>
      <td>4.0000</td>
      <td>3.9500</td>
      <td>4.1900</td>
      <td>4.3000</td>
    </tr>
    <tr>
      <td>6. volume</td>
      <td>5241904</td>
      <td>5449529</td>
      <td>3901401</td>
      <td>4957013</td>
      <td>3598651</td>
      <td>3129214</td>
      <td>4533345</td>
      <td>4298229</td>
      <td>3872433</td>
      <td>3712487</td>
      <td>...</td>
      <td>446900</td>
      <td>304200</td>
      <td>248300</td>
      <td>438700</td>
      <td>521900</td>
      <td>528000</td>
      <td>567700</td>
      <td>1677500</td>
      <td>2221800</td>
      <td>10893600</td>
    </tr>
    <tr>
      <td>7. dividend amount</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <td>8. split coefficient</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>...</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3944 columns</p>
</div>



We need to transpose our table to have proper format for the analysis and forecasting. Rows will be columsn, and columns should be converted to rows.


```python
# saving as csv
# df.to_csv('CRM_stock')
```


```python
# transposing the dataframe
pd.to_datetime(df.columns)
df_transposed = df.transpose().set_index(df.columns)
```


```python
df_transposed.head() # initial look
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. open</th>
      <th>2. high</th>
      <th>3. low</th>
      <th>4. close</th>
      <th>5. adjusted close</th>
      <th>6. volume</th>
      <th>7. dividend amount</th>
      <th>8. split coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-02-21</td>
      <td>191.8400</td>
      <td>192.0400</td>
      <td>186.7200</td>
      <td>189.5000</td>
      <td>189.5000</td>
      <td>5241904</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>2020-02-20</td>
      <td>194.0000</td>
      <td>195.7200</td>
      <td>189.7700</td>
      <td>193.3600</td>
      <td>193.3600</td>
      <td>5449529</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>2020-02-19</td>
      <td>192.0000</td>
      <td>193.9200</td>
      <td>191.8000</td>
      <td>192.8700</td>
      <td>192.8700</td>
      <td>3901401</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>2020-02-18</td>
      <td>190.9500</td>
      <td>191.5000</td>
      <td>188.9200</td>
      <td>191.0900</td>
      <td>191.0900</td>
      <td>4957013</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>2020-02-14</td>
      <td>189.3500</td>
      <td>190.2500</td>
      <td>188.1000</td>
      <td>189.9500</td>
      <td>189.9500</td>
      <td>3598651</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have proper format, we should drop irrelevant columns. For simplicity, I will keep ajusted close, and volume columns.


```python
# removing unused features
df_transposed.drop(columns=['1. open','2. high', '3. low', '4. close','7. dividend amount','8. split coefficient'],inplace=True)
```


```python
df_transposed.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5. adjusted close</th>
      <th>6. volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-02-21</td>
      <td>189.5000</td>
      <td>5241904</td>
    </tr>
    <tr>
      <td>2020-02-20</td>
      <td>193.3600</td>
      <td>5449529</td>
    </tr>
    <tr>
      <td>2020-02-19</td>
      <td>192.8700</td>
      <td>3901401</td>
    </tr>
    <tr>
      <td>2020-02-18</td>
      <td>191.0900</td>
      <td>4957013</td>
    </tr>
    <tr>
      <td>2020-02-14</td>
      <td>189.9500</td>
      <td>3598651</td>
    </tr>
  </tbody>
</table>
</div>



Let's convert data types to float. We can do that using astype method of pandas library.


```python
# converting dtypes to float
df_transposed['5. adjusted close'] = df_transposed['5. adjusted close'].astype(float)
df_transposed['6. volume'] = df_transposed['6. volume'].astype(float)
```


```python
df_transposed.sort_index(ascending=True,inplace=True) # sorting in ascending format
```


```python
df_transposed.index = pd.to_datetime(df_transposed.index) # converting indices to datetime
```


```python
df_transposed.index
```




    DatetimeIndex(['2004-06-23', '2004-06-24', '2004-06-25', '2004-06-28',
                   '2004-06-29', '2004-06-30', '2004-07-01', '2004-07-02',
                   '2004-07-06', '2004-07-07',
                   ...
                   '2020-02-07', '2020-02-10', '2020-02-11', '2020-02-12',
                   '2020-02-13', '2020-02-14', '2020-02-18', '2020-02-19',
                   '2020-02-20', '2020-02-21'],
                  dtype='datetime64[ns]', length=3944, freq=None)



Let's split the data to weekly,monthly, quarterly, and yearly basis. There is no actual purpose for doing  it. I just want to see how the stock price action looks on the chart on various time-frames.


```python
# resampling to weekly, monthly, quarterly and yearly periods

df_weekly = df_transposed.resample('W').mean()
df_monthly = df_transposed.resample('M').mean()
df_quart = df_transposed.resample('Q-DEC').mean()
df_annual = df_transposed.resample('A-DEC').mean()
```


```python
df_weekly.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5. adjusted close</th>
      <th>6. volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2004-06-27</td>
      <td>4.146667</td>
      <td>4.930967e+06</td>
    </tr>
    <tr>
      <td>2004-07-04</td>
      <td>4.074000</td>
      <td>4.609200e+05</td>
    </tr>
    <tr>
      <td>2004-07-11</td>
      <td>4.090625</td>
      <td>3.612750e+05</td>
    </tr>
    <tr>
      <td>2004-07-18</td>
      <td>3.991000</td>
      <td>4.952800e+05</td>
    </tr>
    <tr>
      <td>2004-07-25</td>
      <td>3.489500</td>
      <td>2.005720e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plots

fig = plt.figure(figsize=[15, 7])
plt.suptitle('CRM Price, mean USD', fontsize=18)

plt.subplot(221)
plt.plot(df_weekly['5. adjusted close'], '-', label='By Week')
plt.legend()

plt.subplot(222)
plt.plot(df_monthly['5. adjusted close'], '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(df_quart['5. adjusted close'], '-', label='By Quarter')
plt.legend()

plt.subplot(224)
plt.plot(df_annual['5. adjusted close'], '-', label='By Year')
plt.legend()

# plt.tight_layout()
plt.show()
```


![png](/images/salesforce_timeseries_files/salesforce_timeseries_24_0.png)


We can see on the graph when we take average of larger time periods the line gets smoother and smoother. I will predict monthly stock price of the Salesforce Inc. 


```python
# rename price column
def rename_col(df,col):
    df.rename(columns={col: "Price"},inplace=True)
```


```python
#renaming
rename_col(df_weekly,'5. adjusted close')
rename_col(df_monthly,'5. adjusted close')
rename_col(df_quart,'5. adjusted close')
rename_col(df_annual,'5. adjusted close')
```

# Stationarity Check

In statistic, and in time-series analysis we need to check the data for stationarity. If the data is non-stationary we convert to stationary. 

In the most intuitive sense, <a href= 'https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322'> stationarity </a> means that the statistical properties of a process generating a time series do not change over time. 

Little function below plots seasonal decomposition of a time-series data, and returns p-value of Dickey-Fuller test.


```python
# dickey-fuller test for stationary check
def DF_test(df):
    plt.figure(figsize=[10,5])
    sm.tsa.seasonal_decompose(df.Price).plot()
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df.Price)[1])
    plt.show()
```

## Dickey-Fuller Test

In statistics, the Dickey–Fuller test tests the null hypothesis that a unit root is present in an autoregressive model. The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity. It is named after the statisticians David Dickey and Wayne Fuller, who developed the test in 1979 (<a href='https://en.wikipedia.org/wiki/Dickey–Fuller_test'>Wikipedia</a>)

### Seasonal Decomposition

It is a technique to analyze time-series data. Obeserved data is split into 3 parts: Trend, Seasonal, and Residuals (noise).

* Trend is basically direction of an obeserved data at a time. In our case, the stock price is in up-trend.
* Seasonal data reveals patterns in data that is affected by seasonal factors, and it's repeatly occurs within certain time-frame.
* Residuals is a noise in the data. It is a remainder when trend and seasonality are removed.


Decomposition is primarily used for time series analysis, and as an analysis tool it can be used to inform forecasting models on your problem. It helps to break down your problem and think in structured way.


<a href ='https://en.wikipedia.org/wiki/Decomposition_of_time_series'> Learn more </a>


```python
DF_test(df_monthly)
```

    Dickey–Fuller test: p=1.000000



    <Figure size 720x360 with 0 Axes>



![png](/images/salesforce_timeseries_files/salesforce_timeseries_31_2.png)


Dickey-Fuller test indicates that the data is not stationary as we can see on the graph. P-value is 1.0. We failed to reject null-hypothesis that our data is stationary. P-value should less than or equal to 0.05.

To achieve stationary state we can do various data tranformations. For example, let's try box-cox transformationa and see if we the data stationary...


```python
# Box-Cox Transformations
def box_cox(df):
    df['Price_box'], lmbda = stats.boxcox(df['Price'])
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df['Price_box'])[1])
```


```python
df_monthly['Price_box'], lmbda = stats.boxcox(df_monthly['Price'])
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_monthly['Price_box'])[1])
```

    Dickey–Fuller test: p=0.871254


After Box-Cox transformation our data is still non-stationary. Let's apply log transformation on the data.


```python
# log tranformation
def log_transformation(df):
    df['price_log']=df['Price'].apply(lambda x: np.log(x))
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df['price_log'])[1])
```


```python
# log tranformation
log_transformation(df_monthly)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-08a9f9683a50> in <module>
          1 # log tranformation
    ----> 2 log_transformation(df_monthly)
    

    NameError: name 'log_transformation' is not defined



```python
# Seasonal differentiation
df_monthly['prices_box_diff'] = df_monthly.Price_box - df_monthly.Price_box.shift(12)
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_monthly.prices_box_diff[12:])[1])
```

    Dickey–Fuller test: p=0.003912



```python
sm.tsa.seasonal_decompose(df_monthly['prices_box_diff'][13:]).plot()   
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_monthly['prices_box_diff'][12:])[1])

plt.show()
```

    Dickey–Fuller test: p=0.003912



![png](/images/salesforce_timeseries_files/salesforce_timeseries_41_0.png)


Now our data is stationarized, so we can move to predicting. You can check <a href='https://medium.com/@aabdygaziev/salesforce-inc-stock-price-prediction-time-series-analysis-299bc3f1b631'> Salesforce stock price prediction' </a> on my Medium blog post.


```python

```
