---
layout: post
title: Salesforce Inc. Stock Price Forecasting with Time-Series Analysis
---
***

<a href='https://en.wikipedia.org/wiki/Salesforce'> Salesforce.com, inc.</a> is an American cloud-based software company headquartered in San Francisco, California. It provides customer relationship management (CRM) service and also sells a complementary suite of enterprise applications focused on customer service, marketing automation, analytics, and application development (Wikipedia).

The data set is from <a href= 'https://www.alphavantage.co/#about'> alphavantage </a>. It is very easy to obtain the stock price information from the website. Just follow the documentation.


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



![png](/images/salesforce_timeseries_files/salesforce_timeseries_39_1.png)


* Monthly data is stationarized


```python
# model selection
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_monthly.prices_box_diff[12:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_monthly.prices_box_diff[12:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()
```


![png](/images/salesforce_timeseries_files/salesforce_timeseries_41_0.png)


* Initial parameters according to the graph above:
    - P (0-2)
    - D (0)
    - Q (0-5)
    - seasonality (12)


```python
# fitting to the model
# MA, AR values
p = range(0,2)
d = 1
q = range(0,5)

# seasonal order
P = range(0,2)
D = 1
Q = range(0,2)


parameters = product(p, q, P, Q)
parameters_list = list(parameters)

results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')

for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(df_monthly.Price_box, order=(param[0],d,param[0]), 
                                        seasonal_order=(param[2], d,param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
        results.append([param, model.aic])    

        

```


```python
# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())
```

         parameters         aic
    2  (1, 0, 0, 1) -172.505623
    1  (0, 0, 0, 1) -162.739139
    0  (0, 0, 0, 0)  -62.580627
                                     Statespace Model Results                                 
    ==========================================================================================
    Dep. Variable:                          Price_box   No. Observations:                  189
    Model:             SARIMAX(1, 1, 1)x(0, 1, 1, 12)   Log Likelihood                  90.253
    Date:                            Sun, 23 Feb 2020   AIC                           -172.506
    Time:                                    20:45:55   BIC                           -159.824
    Sample:                                06-30-2004   HQIC                          -167.362
                                         - 02-29-2020                                         
    Covariance Type:                              opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.0548      0.261     -0.210      0.834      -0.566       0.456
    ma.L1          0.3371      0.264      1.277      0.202      -0.180       0.854
    ma.S.L12      -0.9899      1.316     -0.752      0.452      -3.569       1.589
    sigma2         0.0175      0.023      0.778      0.437      -0.027       0.062
    ===================================================================================
    Ljung-Box (Q):                       32.69   Jarque-Bera (JB):                62.34
    Prob(Q):                              0.79   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.40   Skew:                            -0.76
    Prob(H) (two-sided):                  0.00   Kurtosis:                         5.49
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
# STL-decomposition
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[12:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.show()
```

    Dickey–Fuller test:: p=0.000000



![png](/images/salesforce_timeseries_files/salesforce_timeseries_45_1.png)



```python
len(df_monthly)
```




    189




```python
# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))
```


```python
# Prediction

df_month2 = df_monthly[['Price']]

date_list = [datetime(2020, 2, 29),datetime(2020, 3, 30), 
             datetime(2020, 4, 30), datetime(2020, 5, 31), datetime(2020, 6, 30), 
             datetime(2020, 7, 31), datetime(2020, 8, 31), datetime(2020, 9, 30), 
             datetime(2020, 10, 31),datetime(2020, 11, 30)]

future = pd.DataFrame(index=date_list, columns= df_monthly.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = invboxcox((best_model.predict(start=0, end=200)),lmbda)
plt.figure(figsize=(15,7))
df_month2.Price.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Price')
plt.legend()
plt.title('CRM share price, by months')
plt.ylabel('mean USD')
plt.show()
```


![png](/images/salesforce_timeseries_files/salesforce_timeseries_48_0.png)



```python
df_month2.tail(10)
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
      <th>6. volume</th>
      <th>Price</th>
      <th>Price_box</th>
      <th>price_log</th>
      <th>prices_box_diff</th>
      <th>forecast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2020-02-29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>188.345304</td>
    </tr>
    <tr>
      <td>2020-03-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2020-04-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>198.720247</td>
    </tr>
    <tr>
      <td>2020-05-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>202.109957</td>
    </tr>
    <tr>
      <td>2020-06-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>202.907078</td>
    </tr>
    <tr>
      <td>2020-07-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>203.433275</td>
    </tr>
    <tr>
      <td>2020-08-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>204.383844</td>
    </tr>
    <tr>
      <td>2020-09-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>212.845838</td>
    </tr>
    <tr>
      <td>2020-10-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>213.022514</td>
    </tr>
    <tr>
      <td>2020-11-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>217.625027</td>
    </tr>
  </tbody>
</table>
</div>


