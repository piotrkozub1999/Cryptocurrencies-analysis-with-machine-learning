<h1><center>Cryptocurrencies Market Analysis</center></h1>
<h2><center>Machine Learning</center></h2>
<h3><center>EN</center></h3>

## 1. Project objectives

The aim of the project is to conduct research on the behaviour of the cryptocurrency market. The project includes an analysis of the behaviour of the most popular cryptocurrencies on the global market. To this end, a general survey of the cryptocurrency market was conducted, consisting of checking the market capitalisation and share of the 100 most popular cryptocurrencies on the market. An analysis of Bitcoin was also performed as the main cryptocurrency, which has a significant share in the cryptocurrency market and influences the prices of other digital assets. Bitcoin's behaviour was examined in relation to major world events and other interesting factors. Data on various cryptocurrencies was also collated to verify the relationships between them. The project also includes an analysis of real assets on the stock exchange, such as the S&P 500 index, the Dollar index, and the price of gold and silver. Data mining techniques were also used in the analysis to group cryptocurrencies in terms of investment risk.

## 2. Project data

The data used for the analysis was obtained from Binance, the largest cryptocurrency exchange. It collects data from the moment a given cryptocurrency appears on the exchange. This means that the data for individual cryptocurrencies is complete depending on when they appeared on Binance. During the analysis, appropriate time periods were used so that it was possible to download a complete data set for all the necessary cryptocurrencies.  

Access to commodity prices was obtained from data.nasdaq, while the S&P 500 and dollar index quotes were downloaded from Yahoo Finance. Unlike cryptocurrency exchanges, real asset exchanges only operate on business days, which means that there is no data available on non-business days. The solution to this problem is to process the cryptocurrency data appropriately so that it corresponds to the dates of the downloaded stock exchange quotations.

## 3. Tech Stack

The analysis was performed using the Python programming language with ready-made packages enabling data download, processing and visualisation:
* The *Binance* library enables the download of data collected on the Binance website, necessary for the analysis of cryptocurrency exchange rates.
* The *quandl* and *yfinance* libraries are used to download asset data from the stock exchange from websites related to economics and finance.
* The *Pandas*, *NumPy* and *SciPy* libraries enable the appropriate processing of data downloaded from the API. In addition, the *datetime* library allows for the conversion of time data.
* The *Matplotlib* and *seaborn* libraries generate charts based on data. The *mplfinance* library was also used specifically for financial analysis based on cryptocurrency price charts. 
* The *Scikit-Learn* library introduces machine learning and contains various classification, regression and clustering methods.
* The *Statsmodels* library facilitates data exploration and statistical model estimation.


```python
from binance import Client
import quandl
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from  matplotlib.ticker import PercentFormatter
import seaborn as sns
import mplfinance as mpf
from sklearn import metrics
from sklearn.preprocessing import scale
import sklearn.manifold as skm
import sklearn.decomposition as skd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import statsmodels.formula.api as smf
import warnings
```


```python
%matplotlib inline
warnings.filterwarnings("ignore")
```

## 4. Downloading data from the Binance stock market

### 4.1. Access to the Binance exchange


```python
apikey = '...'
secret = '...'
client = Client(apikey, secret)
```

Access to data from the Binance API was obtained using codes that serve as access keys (API key and secret key). The API key is assigned to the *apikey* variable, and the secret key is assigned to the *secret* variable. Both keys are necessary to obtain data from the Binance API and to use and process the data in the further course of the project. Obtaining access keys is not widely available, as in addition to a Binance account, which must be verified with an ID card, you must also have at least PLN 90 invested in cryptocurrencies. Once these criteria are met, it is possible to obtain access keys.

Full access to Binance exchange data was obtained using the *Client* module of the *Binance* library.

### 4.2. Cryptocurrency data retrieval


```python
def data_change(data):
    data_pd = pd.to_datetime(data, format='%d %m %Y')
    data_int = pd.to_datetime(data_pd, unit='s').value
    return int(data_int/1000000)
```

The *data_change* function is responsible for converting a date stored in the ‘day month year’ format into the number of milliseconds that have elapsed since 1 January 1970.


```python
interval = Client.KLINE_INTERVAL_1WEEK
begin = '01 01 2019'
end = '30 11 2022'
begin_int = data_change(begin)
end_int = data_change(end)
```

Variables responsible for retrieving data from the stock exchange have been introduced. The *interval* variable defines the frequency of obtaining data from the stock exchange. The *begin* and *end* variables define the start and end dates for data retrieval. The dates are then converted into the number of milliseconds that have elapsed since 1 January 1970.

In this project, we use data from the beginning of 2019 to the present. We retrieve it at weekly intervals..


```python
def take_coin(slug, symbol, name, ranknow):
    binance = Client(apikey,secret)
    ticker = symbol + "USDT"
    data = binance.get_historical_klines(ticker, interval, start_str=begin_int, end_str=end_int)
    for rows in data:
        rows.append(slug)
        rows.append(symbol)
        rows.append(name)
        rows.append(ranknow)
    return data
```

The *take_coin* function is used to retrieve data for a given cryptocurrency relative to USDT (the digital equivalent of the dollar) based on its name and symbol. New columns have also been added to the retrieved data:
* “slug” - a unique symbol for each cryptocurrency, introduced to fix duplicate tokens shared using a symbol or name,
* “symbol” - the symbol of a given cryptocurrency (ticker),
* “name” - the name of the cryptocurrency,
* “ranknow” - the position in the ranking of the most popular cryptocurrencies


```python
coins_list = []
coins_list.append(take_coin("bitcoin", 'BTC', 'Bitcoin', 1))
coins_list.append(take_coin("ethereum", 'ETH', 'Ethereum', 2))
...
```

```python
data = []
for coins in coins_list:
    data += coins
```

Data collected from the 100 most popular cryptocurrencies based on information from CoinMarketCap. We omit stable coins that simulate the price of the dollar to which we refer.

### 4.3. Data processing


```python
data_df = pd.DataFrame(data)
data_df['Czas otwarcia'] = pd.to_datetime(data_df['Czas otwarcia']/1000, unit='s').dt.date
data_df['Czas zamknięcia'] = pd.to_datetime(data_df['Czas zamknięcia']/1000, unit='s').dt.date
data_df.drop(columns=['Wartość do zignorowania'], axis=1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Opening time</th>
      <th>Opening price</th>
      <th>Highest price during the day</th>
      <th>Lowest price during the day</th>
      <th>Closing price</th>
      <th>Amount of cryptocurrency in circulation</th>
      <th>Closing time</th>
      <th>Change in cryptocurrency transaction volume</th>
      <th>Number of transactions</th>
      <th>Amount of realised volume</th>
      <th>Amount of unrealised volume</th>
      <th>Slug</th>
      <th>Symbol</th>
      <th>Name</th>
      <th>Ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-07</td>
      <td>3987.6200</td>
      <td>4069.8000</td>
      <td>3441.3000</td>
      <td>3476.8100</td>
      <td>2.458873e+05</td>
      <td>2019-01-13</td>
      <td>9.279693e+08</td>
      <td>1671905.0</td>
      <td>1.249704e+05</td>
      <td>4.716823e+08</td>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-14</td>
      <td>3477.5600</td>
      <td>3720.0000</td>
      <td>3467.0200</td>
      <td>3539.2800</td>
      <td>1.993958e+05</td>
      <td>2019-01-20</td>
      <td>7.164520e+08</td>
      <td>1441654.0</td>
      <td>1.039597e+05</td>
      <td>3.735980e+08</td>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-21</td>
      <td>3539.2600</td>
      <td>3662.9400</td>
      <td>3434.8500</td>
      <td>3550.8400</td>
      <td>1.545665e+05</td>
      <td>2019-01-27</td>
      <td>5.491946e+08</td>
      <td>1163997.0</td>
      <td>8.169839e+04</td>
      <td>2.903458e+08</td>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-28</td>
      <td>3550.0500</td>
      <td>3557.7500</td>
      <td>3349.9200</td>
      <td>3458.1100</td>
      <td>1.865741e+05</td>
      <td>2019-02-03</td>
      <td>6.422626e+08</td>
      <td>1288715.0</td>
      <td>9.790823e+04</td>
      <td>3.370756e+08</td>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-02-04</td>
      <td>3458.1100</td>
      <td>3733.5800</td>
      <td>3373.1000</td>
      <td>3680.0600</td>
      <td>1.983507e+05</td>
      <td>2019-02-10</td>
      <td>6.969310e+08</td>
      <td>1313535.0</td>
      <td>1.052102e+05</td>
      <td>3.697665e+08</td>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12392</th>
      <td>2022-10-31</td>
      <td>0.4346</td>
      <td>0.5337</td>
      <td>0.4257</td>
      <td>0.4565</td>
      <td>8.597322e+07</td>
      <td>2022-11-06</td>
      <td>4.115405e+07</td>
      <td>169503.0</td>
      <td>4.097317e+07</td>
      <td>1.960202e+07</td>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>12393</th>
      <td>2022-11-07</td>
      <td>0.4560</td>
      <td>0.4800</td>
      <td>0.2757</td>
      <td>0.2938</td>
      <td>6.168576e+07</td>
      <td>2022-11-13</td>
      <td>2.144259e+07</td>
      <td>127106.0</td>
      <td>3.099059e+07</td>
      <td>1.078844e+07</td>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>12394</th>
      <td>2022-11-14</td>
      <td>0.2938</td>
      <td>0.3437</td>
      <td>0.2739</td>
      <td>0.3105</td>
      <td>5.367182e+07</td>
      <td>2022-11-20</td>
      <td>1.685602e+07</td>
      <td>107327.0</td>
      <td>2.639318e+07</td>
      <td>8.299452e+06</td>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>12395</th>
      <td>2022-11-21</td>
      <td>0.3102</td>
      <td>0.3742</td>
      <td>0.3005</td>
      <td>0.3347</td>
      <td>5.461100e+07</td>
      <td>2022-11-27</td>
      <td>1.848817e+07</td>
      <td>126751.0</td>
      <td>2.613813e+07</td>
      <td>8.850454e+06</td>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>12396</th>
      <td>2022-11-28</td>
      <td>0.3344</td>
      <td>0.3500</td>
      <td>0.3132</td>
      <td>0.3321</td>
      <td>3.061347e+07</td>
      <td>2022-12-04</td>
      <td>1.014330e+07</td>
      <td>62330.0</td>
      <td>1.381859e+07</td>
      <td>4.574167e+06</td>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
<p>12397 rows × 15 columns</p>
</div>



The collected data is divided into appropriate columns. Numerical values stored as *string* are converted to *float* values for correct data analysis in charts. Time values in milliseconds are converted to corresponding dates. The unnecessary column “*Value to be ignored*” is removed.

Based on the downloaded data, we create a new *DataFrame* ‘crypto’ in order to preserve the columns needed for further analysis and improve aesthetics.
At this point, we create several new variables:
- *market* - the capitalisation value of a given cryptocurrency
- *close_ratio* - a measure of the loss in value at the end of each session (the ratio of the maximum price to the closing price)
- *spread* - the amplitude of the value during a single session

### 4.4 Entering logarithmic data

In order to analyse the dynamics of cryptocurrencies over time, the data was modified by:
* introducing a logarithmic scale for the period 2019-2022 due to the high (exponential) rate of change in the value of cryptocurrencies (compared to the traditional stock market),
* inserting a value of “0” in place of any missing data,
* removing incorrectly downloaded rows. A few cryptocurrencies had additional rows, so we only left the rows where the date matched the Bitcoin dates
* adding the variable *birth_time* indicating the age (some cryptocurrencies were created after 2019)
* adding the *log_return* variable indicating the investor's week-to-week rate of return


```python
crypto['log_close'] = np.log(crypto.close)
crypto['log_volume'] = np.log(crypto.volume)
crypto['log_market'] = np.log(crypto.market)
crypto['log_return'] = np.log(crypto.close / crypto.close.shift(1))
```


```python
crypto = crypto.replace([np.inf,-np.inf, np.nan], 0)
btc = crypto[crypto.slug == 'bitcoin']
crypto = crypto.loc[crypto.date.isin(btc.date)]
```


```python
birth_time = pd.DataFrame(crypto.slug.value_counts())
birth_time['survival_time'] = birth_time.slug
birth_time['slug'] = birth_time.index
crypto = pd.merge(crypto, birth_time, how='inner', left_on = 'slug', right_on = 'slug')
```


```python
crypto.head(len(crypto['date']))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>slug</th>
      <th>symbol</th>
      <th>name</th>
      <th>date</th>
      <th>ranknow</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>market</th>
      <th>close_ratio</th>
      <th>spread</th>
      <th>log_close</th>
      <th>log_volume</th>
      <th>log_market</th>
      <th>log_return</th>
      <th>survival_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2019-01-13</td>
      <td>1.0</td>
      <td>3987.6200</td>
      <td>4069.8000</td>
      <td>3441.3000</td>
      <td>3476.8100</td>
      <td>1.249704e+05</td>
      <td>8.549033e+08</td>
      <td>0.056500</td>
      <td>0.180769</td>
      <td>8.153870</td>
      <td>11.735832</td>
      <td>20.566499</td>
      <td>0.000000</td>
      <td>204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2019-01-20</td>
      <td>1.0</td>
      <td>3477.5600</td>
      <td>3720.0000</td>
      <td>3467.0200</td>
      <td>3539.2800</td>
      <td>1.039597e+05</td>
      <td>7.057175e+08</td>
      <td>0.285635</td>
      <td>0.071478</td>
      <td>8.171679</td>
      <td>11.551759</td>
      <td>20.374726</td>
      <td>0.017808</td>
      <td>204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2019-01-27</td>
      <td>1.0</td>
      <td>3539.2600</td>
      <td>3662.9400</td>
      <td>3434.8500</td>
      <td>3550.8400</td>
      <td>8.169839e+04</td>
      <td>5.488407e+08</td>
      <td>0.508527</td>
      <td>0.064236</td>
      <td>8.174939</td>
      <td>11.310790</td>
      <td>20.123319</td>
      <td>0.003261</td>
      <td>204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2019-02-03</td>
      <td>1.0</td>
      <td>3550.0500</td>
      <td>3557.7500</td>
      <td>3349.9200</td>
      <td>3458.1100</td>
      <td>9.790823e+04</td>
      <td>6.451938e+08</td>
      <td>0.520570</td>
      <td>0.060099</td>
      <td>8.148477</td>
      <td>11.491786</td>
      <td>20.285061</td>
      <td>-0.026462</td>
      <td>204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2019-02-10</td>
      <td>1.0</td>
      <td>3458.1100</td>
      <td>3733.5800</td>
      <td>3373.1000</td>
      <td>3680.0600</td>
      <td>1.052102e+05</td>
      <td>7.299424e+08</td>
      <td>0.851531</td>
      <td>0.097955</td>
      <td>8.210684</td>
      <td>11.563715</td>
      <td>20.408476</td>
      <td>0.062207</td>
      <td>204</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12390</th>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>2022-11-06</td>
      <td>100.0</td>
      <td>0.4346</td>
      <td>0.5337</td>
      <td>0.4257</td>
      <td>0.4565</td>
      <td>4.097317e+07</td>
      <td>3.924678e+07</td>
      <td>0.285185</td>
      <td>0.236583</td>
      <td>-0.784167</td>
      <td>17.528428</td>
      <td>17.485380</td>
      <td>0.050083</td>
      <td>123</td>
    </tr>
    <tr>
      <th>12391</th>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>2022-11-13</td>
      <td>100.0</td>
      <td>0.4560</td>
      <td>0.4800</td>
      <td>0.2757</td>
      <td>0.2938</td>
      <td>3.099059e+07</td>
      <td>1.812328e+07</td>
      <td>0.088595</td>
      <td>0.695371</td>
      <td>-1.224856</td>
      <td>17.249194</td>
      <td>16.712708</td>
      <td>-0.440689</td>
      <td>123</td>
    </tr>
    <tr>
      <th>12392</th>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>2022-11-20</td>
      <td>100.0</td>
      <td>0.2938</td>
      <td>0.3437</td>
      <td>0.2739</td>
      <td>0.3105</td>
      <td>2.639318e+07</td>
      <td>1.666510e+07</td>
      <td>0.524355</td>
      <td>0.224799</td>
      <td>-1.169571</td>
      <td>17.088616</td>
      <td>16.628827</td>
      <td>0.055285</td>
      <td>123</td>
    </tr>
    <tr>
      <th>12393</th>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>2022-11-27</td>
      <td>100.0</td>
      <td>0.3102</td>
      <td>0.3742</td>
      <td>0.3005</td>
      <td>0.3347</td>
      <td>2.613813e+07</td>
      <td>1.827830e+07</td>
      <td>0.464043</td>
      <td>0.220197</td>
      <td>-1.094521</td>
      <td>17.078906</td>
      <td>16.721225</td>
      <td>0.075051</td>
      <td>123</td>
    </tr>
    <tr>
      <th>12394</th>
      <td>storj</td>
      <td>STORJ</td>
      <td>Storj</td>
      <td>2022-12-04</td>
      <td>100.0</td>
      <td>0.3344</td>
      <td>0.3500</td>
      <td>0.3132</td>
      <td>0.3321</td>
      <td>1.381859e+07</td>
      <td>1.016673e+07</td>
      <td>0.513587</td>
      <td>0.110810</td>
      <td>-1.102319</td>
      <td>16.441525</td>
      <td>16.134632</td>
      <td>-0.007798</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
<p>12395 rows × 18 columns</p>
</div>



## 5. Cryptocurrency market overview


```python
crypto_market = crypto.groupby('date')[['market']].sum()
```

Grouping and summing data by date for the “*market*” and “*volume*” columns to obtain information about the market size and volume of cryptocurrencies on a given day.


```python
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title('Całkowita wartość rynku')
plt.xlabel('Data')
plt.ylabel('Wartość rynku')
sns.lineplot(data=crypto_market.market, color="#0000ac", label='Market capitalisation')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_37_0.png)
    


In the chart above, we can see that the value of the cryptocurrency market was highest in mid-2021. The rapid growth was followed by an equally rapid market collapse.

In order to show the current distribution of market capitalisation on the Binance exchange, we will compare its values for the last day of our time series, i.e. 12 April 2022.
The comparison will be visualised in a pie chart for the 10 largest cryptocurrencies.


```python
crypto_snap['composition'] = np.where(crypto_snap.ranknow <= 10, crypto_snap.ranknow, 11)
crypto_market_comp = crypto_snap.groupby(by = ['composition'])['market'].sum()
```

Creation of eleven compartments, which will contain the ten most popular cryptocurrencies and the total value from the eleventh to the hundredth cryptocurrency.


```python
labels = crypto_snap[crypto_snap.ranknow <= 11].name.replace('Uniswap', 'Pozostałe kryptowaluty')
sizes = crypto_market_comp
explode = (0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize=(8,8))
ax.set_title('Udział największych kryptowalut w całym rynku', fontsize=20)
ax.pie(sizes, explode = explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={"fontsize":14}, 
        labeldistance=1.2, pctdistance=0.9)
ax.axis('equal')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_43_0.png)
    


As we can see, Bitcoin accounts for the vast majority of the market value. Ethereum is worth less than half of that, while the rest accounts for 25% of the capitalisation.


```python
crypto_4 = crypto[(crypto.ranknow <= 4)]
```

## 6. Analysis of the most popular cryptocurrencies

When analysing the most popular cryptocurrencies since the beginning of 2019, Bitcoin and Ethereum have shown the greatest stability, always occupying first and second place respectively during the given time period. For most of the time, until the beginning of 2021, the third place belonged to Ripple, but the rise in popularity of the Binance exchange caused their cryptocurrency to gain in value and since 2021 it has been consistently the third most popular cryptocurrency. During this time, Ripple hovered in the top 6 popular cryptocurrencies.  
It can therefore be concluded that over the last 4 years, the most stable cryptocurrencies were Bitcoin, Ethereum, BinanceCoin and Ripple.



```python
crypto_4 = crypto[(crypto.ranknow <= 4)]
```


```python
fig = sns.relplot(x = 'date', y = 'log_market', kind = 'line', data = crypto_4, hue = 'name')
fig.fig.set_figwidth(9)
fig.ax.set_title("Top 4 stablinych kryptowalut")
plt.xlabel('Data')
plt.ylabel('Wartość rynku (logarytmiczna)')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_49_0.png)
    


The chart above shows the market value of the TOP 4 cryptocurrencies on the Binance exchange. However, the logarithmic scale does not reflect Bitcoin's actual advantage over the rest.
On the other hand, not using a logarithmic scale will cause the values of other cryptocurrencies to approach the x-axis.


```python
sns.pairplot(crypto_4[['name','log_market','log_volume','log_return','log_close','spread','close_ratio']], hue='name')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_51_0.png)
    


The above set of charts shows the relationships between five values (market value, realised volume, rate of return, closing ratio, weekly amplitude and closing value) for the four largest cryptocurrencies.
By observing the individual relationships, we can see that the combinations of charts related to realised volume and value are separate. In most cases, these sets can be easily separated for individual cryptocurrencies. The remaining variables generally overlap.


```python
fig, ax = plt.subplots(figsize= (12,8))
sns.boxplot(x='name', y='log_return', data = crypto[crypto.ranknow <= 10], ax = ax)
ax.set_title('Spektrum wydajności dla 10 najpopularniejszych kryptowalut')
ax.set_xlabel('')
ax.set_ylabel('Wartość logartymiczna stopy zwrotu')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_53_0.png)
    


The box plot above shows the distribution of returns for individual cryptocurrencies.
The most obvious conclusion can be drawn from the distribution of outliers. We can see that most cryptocurrencies, unlike Bitcoin, have isolated cases of sharp week-to-week declines. This means that purchasing them carries a low but possible risk of a sharp decline in value.

## 7. Bitcoin Analysis


```python
bitcoin = crypto[crypto.slug == 'bitcoin']
bitcoin['date'] = pd.to_datetime(bitcoin['date'])
```

Creating a variable responsible for all data related only to Bitcoin


```python
mpf.plot(bitcoin.set_index('date').tail(len(bitcoin)),
        figscale=1.5, type='candle', style='yahoo',
        volume=True, title='Wykres Bitcoina do USDT', mav=(5))
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_58_0.png)
    


The chart above shows the price of Bitcoin against the cryptocurrency Tether, which is the digital equivalent of the dollar and fluctuates around one dollar per token.  

Since 2019, large American corporations have begun to accept a new method of payment using the digital currency Bitcoin. This caused the price to rise to £10,000 and then fluctuate around £8,500. At the end of 2019, the Covid pandemic broke out in Asia, which translated into another increase in the cryptocurrency's price. The reason was the onset of a crisis in countries opposed to cryptocurrencies as a form of payment (e.g. China). When the pandemic reached Europe and the US, the price of Bitcoin fell sharply and tokens were largely sold off. Under the influence of economic instability, the value of the cryptocurrency began to rise again, reaching the value achieved before the pandemic (around £1,000) and stabilising at that value for over six months.  

At the end of 2020, Bitcoin experienced its largest price increase in history, reaching a value of over $61,000. The value changed by over 500% in 4 months. The reason for the large increase in cryptocurrency prices was the effects of the crisis caused by the pandemic. People began to look for solutions to the difficult economic situation, and more and more companies, such as Amazon, Bank of New York and Morgan Stanley, allowed transactions to be made using Bitcoin. Influential people, such as Elon Musk, began to invest in BTC tokens and spread information about them, which also caused the price to rise. At one point, after the cryptocurrency's highest value stabilised, there was general panic and a sudden mass sell-off of the cryptocurrency by short-term investors. The result was a sudden drop in the price of over 40%. The new value of BTC was around $33,000. After the mass panic, as always, there was a period of growth in value, which was halted by the Chinese government. In September 2021, the People's Bank of China introduced a ban on cryptocurrency transactions, which affected Bitcoin's stock market prices. The growth was halted, but only for a short period, after which the value of Bitcoin reached a record high of over $64,000 per 1 BTC. This is the highest value of Bitcoin in its history.

At the end of 2021, Bitcoin began to decline in value due to a correction in prices. After a long bull market in 2021, it was time for a bear market. The decline continued until February of this year, when the value of the cryptocurrency reached large fluctuations around $40,000. Under the influence of the outbreak of war between Russia and Ukraine, the value of Bitcoin initially rose, but due to high inflation and economic uncertainty, the price began to fall again. Another factor influencing the price was the decreasing number of tokens available for mining, and thus the increasingly difficult process. Since June, the price of Bitcoin has stabilised and is fluctuating around $20,000 with a slight downward trend.  

The BTC token is closely linked to the current political and economic situation. The exchange rate of cryptocurrencies is greatly influenced by both countries and their decisions, as well as large companies listed on the stock exchange.  The value of Bitcoin can be easily manipulated depending on the decisions
made by high-ranking individuals, which has a negative impact on the independence of cryptocurrencies.



```python
bitcoin.describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ranknow</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>market</th>
      <th>close_ratio</th>
      <th>spread</th>
      <th>log_close</th>
      <th>log_volume</th>
      <th>log_market</th>
      <th>log_return</th>
      <th>survival_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>204.0</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>2.040000e+02</td>
      <td>2.040000e+02</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>23659.739510</td>
      <td>25486.211961</td>
      <td>21731.535539</td>
      <td>23724.031520</td>
      <td>2.789147e+05</td>
      <td>1.308439e+10</td>
      <td>0.545250</td>
      <td>0.150576</td>
      <td>9.760339</td>
      <td>12.314905</td>
      <td>22.774710</td>
      <td>0.007810</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>17765.508196</td>
      <td>19030.397275</td>
      <td>16134.540639</td>
      <td>17717.528633</td>
      <td>2.397284e+05</td>
      <td>1.188581e+10</td>
      <td>0.272440</td>
      <td>0.091156</td>
      <td>0.827828</td>
      <td>0.610240</td>
      <td>1.135906</td>
      <td>0.096148</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>3458.110000</td>
      <td>3557.750000</td>
      <td>3349.920000</td>
      <td>3458.110000</td>
      <td>8.169839e+04</td>
      <td>5.488407e+08</td>
      <td>0.002791</td>
      <td>0.032204</td>
      <td>8.148477</td>
      <td>11.310790</td>
      <td>20.123319</td>
      <td>-0.404390</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>9018.370000</td>
      <td>9586.250000</td>
      <td>8504.000000</td>
      <td>9061.925000</td>
      <td>1.456269e+05</td>
      <td>3.197841e+09</td>
      <td>0.305517</td>
      <td>0.092451</td>
      <td>9.111836</td>
      <td>11.888800</td>
      <td>21.885688</td>
      <td>-0.027063</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>18299.440000</td>
      <td>19452.560000</td>
      <td>16091.635000</td>
      <td>18299.710000</td>
      <td>1.920382e+05</td>
      <td>1.028342e+10</td>
      <td>0.605052</td>
      <td>0.123094</td>
      <td>9.814621</td>
      <td>12.165423</td>
      <td>23.053799</td>
      <td>0.013020</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>38395.367500</td>
      <td>42403.677500</td>
      <td>34543.155000</td>
      <td>38395.370000</td>
      <td>2.929799e+05</td>
      <td>2.063546e+10</td>
      <td>0.767028</td>
      <td>0.184464</td>
      <td>10.555692</td>
      <td>12.587842</td>
      <td>23.750277</td>
      <td>0.062301</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>65519.110000</td>
      <td>69000.000000</td>
      <td>62278.000000</td>
      <td>65519.100000</td>
      <td>1.600578e+06</td>
      <td>5.292208e+10</td>
      <td>0.997225</td>
      <td>0.820170</td>
      <td>11.090097</td>
      <td>14.285875</td>
      <td>24.692087</td>
      <td>0.230975</td>
      <td>204.0</td>
    </tr>
  </tbody>
</table>
</div>



Display statistics for each Bitcoin data column:
* “count” - number of rows, sum of the lengths of all time series
* “mean” - mean value
* “std” - standard deviation value
* “min” - minimum value
* “25%” - lower quartile, twenty-fifth percentile
* “50%” - median value, 50th percentile
* “75%” - upper quartile, 75th percentile
* “max” - maximum value

The above table with statistics allows for a more accurate analysis of the Bitcoin exchange rate. The minimum value reached by Bitcoin since the beginning of 2019 is $3,349.92, while the maximum is $69,000. The standard deviation is very high, which means that there is a large variation in the cryptocurrency's exchange rate. The percentile values confirm that most of the time, the values were lower. The average value is around $23,700, while the median is just under $18,300. There is a fairly large difference between these values, which confirms the high volatility of the exchange rate and the short periods of time when Bitcoin reached its highest values.


```python
def plot_time_series(time_series, crypto_title):
    fig, (histogram, probability) = plt.subplots(1, 2, figsize=(12, 4))
    histogram.hist(time_series, bins = 20)
    histogram.set_title('Rozkład szeregów czasowych' + ' ' + crypto_title)
    histogram.set_xlabel('stopa zwortu (logarytm)')
    histogram.set_ylabel('częstotliwość')
    stats.probplot(time_series, dist='norm', plot=plt)
    probability.set_title('Wykres prawdopodobieństwa' + ' ' + crypto_title)
    probability.set_xlabel('kwantyle teoretyczne')
    probability.set_ylabel('uporządkowane wartości')
    plt.show()
```

The *plot_time_series* function is used to create a graph of the distribution of time series for a given cryptocurrency and a probability graph for the purpose of analysing individual cryptocurrencies.


```python
plot_time_series(bitcoin.log_return, 'Bitcoina')
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_65_0.png)
    



```python
bitcoin_stat = [bitcoin.log_return.mean(), bitcoin.log_return.std(), bitcoin.log_return.skew(), bitcoin.log_return.kurtosis()]
pd.DataFrame(bitcoin_stat, columns = list(["Statystyki Bitcoina"]), 
             index = list(["Średnia arytmetyczna", "Odchylenie standardowe", "Skośność", "Kurtoza"]))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Statystyki Bitcoina</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Średnia arytmetyczna</th>
      <td>0.007810</td>
    </tr>
    <tr>
      <th>Odchylenie standardowe</th>
      <td>0.096148</td>
    </tr>
    <tr>
      <th>Skośność</th>
      <td>-0.701549</td>
    </tr>
    <tr>
      <th>Kurtoza</th>
      <td>1.893684</td>
    </tr>
  </tbody>
</table>
</div>



The calculated statistics for Bitcoin's rate of return show that the standard deviation is high in relation to the mean value.
A skewness of -0.7 indicates a left-sided tail of the histogram, which can be observed in the above graph of the rate of return distribution.
Positive kurtosis indicates a greater number of positive outliers, which is also shown in the chart (higher bars with positive values).
Both of these statistics indicate that the return distribution is far from normal.


## 8. Analysis of cryptocurrencies in relation to assets on the stock exchange

The aim of this part of the analysis is to compare digital assets with real assets on stock exchanges in order to find correlations between individual assets. From the previously presented 100 most popular cryptocurrencies, Bitcoin, Ethereum, Binance Coin and Ripple were selected. In the case of stock market assets, the prices of commodities such as gold and silver were selected. The S&P 500 index and the dollar index, i.e. the ratio of the US dollar to a basket of foreign currencies, were also selected.

### 8.1. Downloading data from the cryptocurrency exchange


```python
interval = Client.KLINE_INTERVAL_1DAY
begin_int = data_change('01 01 2022')
end_int = data_change('10 12 2022')
```

In order to perform a more accurate analysis, it was decided to change the dates to the period from the beginning to the end of 2022. The frequency of data collection has also changed. The data is updated daily rather than weekly, as was the case during the first analysis.


```python
day_coins_list = []
day_coins_list.append(take_coin("bitcoin", 'BTC', 'Bitcoin', 1))
day_coins_list.append(take_coin("ethereum", 'ETH', 'Ethereum', 2))
day_coins_list.append(take_coin("binancecoin", 'BNB', 'Binance Coin', 3))
day_coins_list.append(take_coin("ripple", 'XRP', 'Ripple', 4))
```


```python
day_data = []
for day_coins in day_coins_list:
    day_data += day_coins
```


```python
data_df = pd.DataFrame(day_data)
data_df.columns = columns_name
data_df.drop(columns=['Wartość do zignorowania'], axis=1)
data_df[numeric_columns] = data_df[numeric_columns].apply(pd.to_numeric, axis=1)
data_df['Czas otwarcia'] = pd.to_datetime(data_df['Czas otwarcia']/1000, unit='s').dt.date
data_df['Czas zamknięcia'] = pd.to_datetime(data_df['Czas zamknięcia']/1000, unit='s').dt.date
```


Creating a new table with processed personalised data that will be used during the analysis.


```python
crypto2022['log_close'] = np.log(crypto2022.close)
crypto2022['log_volume'] = np.log(crypto2022.volume)
crypto2022['log_market'] = np.log(crypto2022.market)
crypto2022['spread'] = (crypto2022.high - crypto2022.low) / crypto2022.close
crypto2022['log_return'] = np.log(crypto2022.close / crypto2022.close.shift(1))
crypto2022 = crypto2022.replace([np.inf,-np.inf, np.nan], 0)
```

Introduction of logarithmic values identical to those used in the previous analysis. Missing values were also replaced with a value equal to 0.


```python
bitcoin2022 = crypto2022[crypto2022.slug == 'bitcoin']
ethereum2022 = crypto2022[crypto2022.slug == 'ethereum']
binancecoin2022 = crypto2022[crypto2022.slug == 'binancecoin']
ripple2022 = crypto2022[crypto2022.slug == 'ripple']
```


```python
crypto_asset = pd.DataFrame(columns =[], index = bitcoin2022.date)
crypto_asset = pd.merge(crypto_asset, bitcoin2022[['date', 'log_return']], 
                        how='inner', left_on = 'date', right_on = 'date')
crypto_asset = pd.merge(crypto_asset, ethereum2022[['date', 'log_return']], 
                        how='inner', left_on = 'date', right_on = 'date')
crypto_asset = pd.merge(crypto_asset, binancecoin2022[['date', 'log_return']], 
                        how='inner', left_on = 'date', right_on = 'date')
crypto_asset = pd.merge(crypto_asset, ripple2022[['date', 'log_return']], 
                        how='inner', left_on = 'date', right_on = 'date')
crypto_asset.columns = ['Date','Bitcoin', 'Ethereum', 'Binance Coin', 'Ripple']
crypto_asset = crypto_asset.set_index('Date')
crypto_asset.index = pd.to_datetime(crypto_asset.index)
```

Creation of a table containing the date and logarithmic return value for the previously specified cryptocurrencies. The data contained in the table has been processed accordingly.


```python
crypto_asset.head(len(crypto_asset))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bitcoin</th>
      <th>Ethereum</th>
      <th>Binance Coin</th>
      <th>Ripple</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-01</th>
      <td>0.000000</td>
      <td>-1.514793</td>
      <td>-0.876124</td>
      <td>-5.825350</td>
    </tr>
    <tr>
      <th>2022-01-02</th>
      <td>-0.009188</td>
      <td>0.016522</td>
      <td>0.006992</td>
      <td>0.009244</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>-0.017926</td>
      <td>-0.016429</td>
      <td>-0.036633</td>
      <td>-0.030990</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>-0.013310</td>
      <td>0.005091</td>
      <td>-0.009816</td>
      <td>-0.011600</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>-0.053346</td>
      <td>-0.066770</td>
      <td>-0.066895</td>
      <td>-0.062952</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-12-06</th>
      <td>0.007201</td>
      <td>0.009412</td>
      <td>0.006222</td>
      <td>0.005380</td>
    </tr>
    <tr>
      <th>2022-12-07</th>
      <td>-0.014875</td>
      <td>-0.032083</td>
      <td>-0.020892</td>
      <td>-0.020390</td>
    </tr>
    <tr>
      <th>2022-12-08</th>
      <td>0.022752</td>
      <td>0.039028</td>
      <td>0.021581</td>
      <td>0.027265</td>
    </tr>
    <tr>
      <th>2022-12-09</th>
      <td>-0.005562</td>
      <td>-0.013432</td>
      <td>-0.015267</td>
      <td>-0.013282</td>
    </tr>
    <tr>
      <th>2022-12-10</th>
      <td>-0.000062</td>
      <td>0.002570</td>
      <td>0.007663</td>
      <td>-0.006190</td>
    </tr>
  </tbody>
</table>
<p>344 rows × 4 columns</p>
</div>



### 8.2. Downloading real asset data from the stock exchange


```python
quandl.ApiConfig.api_key = “...”
```

Access to data contained on the data.nasdaq website (formerly quandl) is obtained through a special access key (*api_key*).


```python
start = datetime(2022, 1, 1)
end = datetime(2022, 12, 10)
```

Enter the time range specified in the previous section, in which data for three cryptocurrencies was also retrieved from the beginning of 2022.


```python
gold_price = quandl.get("LBMA/GOLD", start_date = start, end_date = end)
silver_price = quandl.get("LBMA/SILVER", start_date = start, end_date = end)
stock_index = yf.download("^GSPC", start, end)
USD_index = yf.download('DX-Y.NYB', start, end)
```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    

Data on gold and silver prices relative to various currencies within a predefined time frame was obtained from data.nasdaq. Data on the S&P 500 index and the dollar index was also obtained from Yahoo Finance.


```python
gold_price.index = gold_price.index.date
silver_price.index = silver_price.index.date
stock_index.index = stock_index.index.date
stock_index.drop(index=stock_index.index[0], axis=0, inplace=True)
USD_index.index = USD_index.index.date
USD_index.drop(index=USD_index.index[0], axis=0, inplace=True)
```


```python
stock_market_asset = pd.merge(gold_price, silver_price, left_index=True, right_index=True)
stock_market_asset = pd.merge(stock_market_asset, stock_index, left_index=True, right_index=True)
stock_market_asset = pd.merge(stock_market_asset, USD_index, left_index=True, right_index=True)
stock_market_asset = stock_market_asset[['USD (AM)', 'USD',  'Adj Close_x', 'Adj Close_y']]
stock_market_asset.columns = ['Kurs złota', 'Kurs srebra', 'Kurs S&P 500', 'Kurs USD']
```

Combination of previously downloaded stock market data into a table. Only the closing price for each day is retained.


```python
stock_market_asset.head(len(stock_market_asset))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kurs złota</th>
      <th>Kurs srebra</th>
      <th>Kurs S&amp;P 500</th>
      <th>Kurs USD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-04</th>
      <td>1809.05</td>
      <td>22.890</td>
      <td>4793.540039</td>
      <td>96.290001</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>1818.50</td>
      <td>23.055</td>
      <td>4700.580078</td>
      <td>96.190002</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>1804.95</td>
      <td>22.245</td>
      <td>4696.049805</td>
      <td>96.250000</td>
    </tr>
    <tr>
      <th>2022-01-07</th>
      <td>1792.20</td>
      <td>22.240</td>
      <td>4677.029785</td>
      <td>95.739998</td>
    </tr>
    <tr>
      <th>2022-01-10</th>
      <td>1800.55</td>
      <td>22.455</td>
      <td>4670.290039</td>
      <td>95.989998</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-12-05</th>
      <td>1794.35</td>
      <td>22.985</td>
      <td>3998.840088</td>
      <td>105.290001</td>
    </tr>
    <tr>
      <th>2022-12-06</th>
      <td>1773.35</td>
      <td>22.540</td>
      <td>3941.260010</td>
      <td>105.580002</td>
    </tr>
    <tr>
      <th>2022-12-07</th>
      <td>1771.85</td>
      <td>22.385</td>
      <td>3933.919922</td>
      <td>105.099998</td>
    </tr>
    <tr>
      <th>2022-12-08</th>
      <td>1782.45</td>
      <td>22.695</td>
      <td>3963.510010</td>
      <td>104.769997</td>
    </tr>
    <tr>
      <th>2022-12-09</th>
      <td>1793.00</td>
      <td>23.110</td>
      <td>3934.379883</td>
      <td>104.809998</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 4 columns</p>
</div>



All prices are then converted into a logarithmic rate of return.


```python
for col in stock_market_asset.columns:
    stock_market_asset[col] = np.log(stock_market_asset[col] / stock_market_asset[col].shift(1))
stock_market_asset.head(len(stock_market_asset))
stock_market_asset.drop(index=stock_market_asset.index[0], axis=0, inplace=True)
```


Only common dates have been retained so that the number of rows matches. This is necessary because the stock exchange only operates on working days, while the cryptocurrency exchange is active every day.


```python
eight_assets = pd.merge(crypto_asset, stock_market_asset, left_index=True, right_index=True)
eight_assets.head(len(eight_assets))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bitcoin</th>
      <th>Ethereum</th>
      <th>Binance Coin</th>
      <th>Ripple</th>
      <th>Kurs złota</th>
      <th>Kurs srebra</th>
      <th>Kurs S&amp;P 500</th>
      <th>Kurs USD</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-05</th>
      <td>-0.053346</td>
      <td>-0.066770</td>
      <td>-0.066895</td>
      <td>-0.062952</td>
      <td>0.005210</td>
      <td>0.007183</td>
      <td>-0.019583</td>
      <td>-0.001039</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>-0.008524</td>
      <td>-0.038528</td>
      <td>-0.003169</td>
      <td>0.008378</td>
      <td>-0.007479</td>
      <td>-0.035765</td>
      <td>-0.000964</td>
      <td>0.000624</td>
    </tr>
    <tr>
      <th>2022-01-07</th>
      <td>-0.035818</td>
      <td>-0.062863</td>
      <td>-0.053233</td>
      <td>-0.020357</td>
      <td>-0.007089</td>
      <td>-0.000225</td>
      <td>-0.004058</td>
      <td>-0.005313</td>
    </tr>
    <tr>
      <th>2022-01-10</th>
      <td>-0.001007</td>
      <td>-0.022228</td>
      <td>-0.032433</td>
      <td>-0.017550</td>
      <td>0.004648</td>
      <td>0.009621</td>
      <td>-0.001442</td>
      <td>0.002608</td>
    </tr>
    <tr>
      <th>2022-01-11</th>
      <td>0.021450</td>
      <td>0.049688</td>
      <td>0.087639</td>
      <td>0.040265</td>
      <td>0.002579</td>
      <td>0.005994</td>
      <td>0.009118</td>
      <td>-0.003862</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-12-05</th>
      <td>-0.008180</td>
      <td>-0.015756</td>
      <td>-0.011720</td>
      <td>-0.000514</td>
      <td>-0.003560</td>
      <td>0.016671</td>
      <td>-0.018056</td>
      <td>0.007053</td>
    </tr>
    <tr>
      <th>2022-12-06</th>
      <td>0.007201</td>
      <td>0.009412</td>
      <td>0.006222</td>
      <td>0.005380</td>
      <td>-0.011772</td>
      <td>-0.019550</td>
      <td>-0.014504</td>
      <td>0.002751</td>
    </tr>
    <tr>
      <th>2022-12-07</th>
      <td>-0.014875</td>
      <td>-0.032083</td>
      <td>-0.020892</td>
      <td>-0.020390</td>
      <td>-0.000846</td>
      <td>-0.006900</td>
      <td>-0.001864</td>
      <td>-0.004557</td>
    </tr>
    <tr>
      <th>2022-12-08</th>
      <td>0.022752</td>
      <td>0.039028</td>
      <td>0.021581</td>
      <td>0.027265</td>
      <td>0.005965</td>
      <td>0.013754</td>
      <td>0.007494</td>
      <td>-0.003145</td>
    </tr>
    <tr>
      <th>2022-12-09</th>
      <td>-0.005562</td>
      <td>-0.013432</td>
      <td>-0.015267</td>
      <td>-0.013282</td>
      <td>0.005901</td>
      <td>0.018121</td>
      <td>-0.007377</td>
      <td>0.000382</td>
    </tr>
  </tbody>
</table>
<p>229 rows × 8 columns</p>
</div>



Combining a table with cryptocurrency data and stock market quotes in order to analyse the correlation between these quotes.

### 8.3. Correlation analysis


```python
corr_matrix = eight_assets.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_110_0.png)
    


The chart above shows the correlation matrix of the rates of return for the assets described above. It can be seen that the correlation between all cryptocurrencies is very strong, ranging from a minimum of 0.76 to a maximum of 0.9 (excluding the correlation between the same assets).  
In the case of commodities, the same relationships exist as between cryptocurrencies, which means that the prices of gold and silver are strongly correlated. If we check the impact of cryptocurrencies on these commodities, the correlation is close to 0, which means that there is virtually no correlation between them.  
The third real asset is the S&P 500 index, which shows a moderate correlation with cryptocurrencies. Bitcoin and Ethereum, as the most influential cryptocurrencies on the market, have the highest correlation. The correlation between the S&P 500 index and the value of gold and silver is negligible.
The last asset analysed is the dollar index. In all cases, the dollar index is correlated with other assets as the main currency of transactions in the world. Importantly, the correlation in all cases is negative, i.e. an increase in the value of the dollar causes a decrease in the value of other assets. The correlation between the dollar and cryptocurrencies and commodities is weak, at around -0.3 and -0.25, respectively. The correlation with the S&P 500 index is -0.48, which confirms the correctness of the data processing.

## 9. Application of selected machine learning algorithms

In this chapter, a base model will be created to group all analysed cryptocurrencies. After appropriate data preparation, the following will be performed:
- dimensionality reduction using the PCA method
- clustering using the k-means method
- evaluation of models using the following coefficients: Silhouette Score and Caliński-Harabasz index
- presentation of clustering results

The expected result of clustering is the grouping of cryptocurrencies, which will make it possible to assess which ones are worth investing in and which ones are better to avoid.

### 9.1. Data preparation


```python
crypto_mean = pd.DataFrame(crypto.groupby(“slug”).mean())
```

The data used in clustering cannot take the form of time series. From now on, each cryptocurrency will have one value in each column. For this purpose, the average value for all components of a given cryptocurrency was calculated.


```python
crypto_log_return = crypto.groupby('slug').log_return
ret_std = pd.DataFrame(crypto_log_return.std())
ret_skew = pd.DataFrame(crypto_log_return.skew())
ret_kurt = pd.DataFrame(crypto_log_return.apply(pd.DataFrame.kurt))

ret_std.columns = ['std']
ret_skew.columns = ['skew']
ret_kurt.columns = ['kurt']
```

In addition, additional indicators based on the average logarithmic return rate of each cryptocurrency were calculated:
- standard deviation
- skewness
- kurtosis


```python
df = pd.merge(crypto_mean, ret_std, on = 'slug')
df = pd.merge(df, ret_skew, on = 'slug')
df = pd.merge(df, ret_kurt, on = 'slug')
df.dropna(inplace = True)
```


Selecting data for clustering:


```python
df = df[['close_ratio', 'spread', 'log_market', 'log_volume', 'log_close', 'log_return', 'std', 'skew', 'kurt']]
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>close_ratio</th>
      <th>spread</th>
      <th>log_market</th>
      <th>log_volume</th>
      <th>log_close</th>
      <th>log_return</th>
      <th>std</th>
      <th>skew</th>
      <th>kurt</th>
    </tr>
    <tr>
      <th>slug</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1inchnetwork</th>
      <td>0.459766</td>
      <td>0.328432</td>
      <td>18.588021</td>
      <td>17.300170</td>
      <td>0.579122</td>
      <td>-0.012430</td>
      <td>0.193209</td>
      <td>-0.230027</td>
      <td>3.473053</td>
    </tr>
    <tr>
      <th>aave</th>
      <td>0.479666</td>
      <td>0.308444</td>
      <td>19.233919</td>
      <td>13.436724</td>
      <td>5.099144</td>
      <td>0.053185</td>
      <td>0.552261</td>
      <td>8.763503</td>
      <td>86.893004</td>
    </tr>
    <tr>
      <th>algorand</th>
      <td>0.453142</td>
      <td>0.283847</td>
      <td>18.278901</td>
      <td>18.221501</td>
      <td>-0.661236</td>
      <td>0.005546</td>
      <td>0.288618</td>
      <td>6.802630</td>
      <td>72.888298</td>
    </tr>
    <tr>
      <th>amp</th>
      <td>0.278006</td>
      <td>0.226610</td>
      <td>16.241162</td>
      <td>19.842580</td>
      <td>-4.308273</td>
      <td>-0.110280</td>
      <td>0.418625</td>
      <td>-6.814104</td>
      <td>48.606836</td>
    </tr>
    <tr>
      <th>ankr</th>
      <td>0.439980</td>
      <td>0.325128</td>
      <td>16.783145</td>
      <td>20.144555</td>
      <td>-4.075641</td>
      <td>-0.024612</td>
      <td>0.469527</td>
      <td>-9.904553</td>
      <td>120.315089</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>waves</th>
      <td>0.449091</td>
      <td>0.289203</td>
      <td>17.509529</td>
      <td>15.357694</td>
      <td>1.446488</td>
      <td>-0.002412</td>
      <td>0.189623</td>
      <td>-0.019559</td>
      <td>3.434542</td>
    </tr>
    <tr>
      <th>wax</th>
      <td>0.447308</td>
      <td>0.297511</td>
      <td>17.395406</td>
      <td>18.353696</td>
      <td>-1.642644</td>
      <td>-0.077713</td>
      <td>0.445931</td>
      <td>-6.470252</td>
      <td>48.557921</td>
    </tr>
    <tr>
      <th>winklink</th>
      <td>0.404516</td>
      <td>0.279241</td>
      <td>17.415594</td>
      <td>25.343371</td>
      <td>-8.588089</td>
      <td>-0.077344</td>
      <td>0.940121</td>
      <td>-12.192324</td>
      <td>157.671522</td>
    </tr>
    <tr>
      <th>woonetwork</th>
      <td>0.400365</td>
      <td>0.300712</td>
      <td>17.377150</td>
      <td>18.145546</td>
      <td>-1.501281</td>
      <td>-0.027472</td>
      <td>0.170444</td>
      <td>0.018143</td>
      <td>-0.039457</td>
    </tr>
    <tr>
      <th>zilliqa</th>
      <td>0.471734</td>
      <td>0.291522</td>
      <td>17.553460</td>
      <td>20.371173</td>
      <td>-3.545177</td>
      <td>-0.015618</td>
      <td>0.307639</td>
      <td>-6.701898</td>
      <td>75.027918</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 9 columns</p>
</div>



The clustering model cannot include all of our columns, as some of them have no impact on the profitability of a cryptocurrency. Analysing data such as intraday prices or cryptocurrency rankings would unnecessarily distort the results.
For this purpose, only the following parameters will be retained:
- closing price to opening price ratio
- spread
- logarithm of market capitalisation
- logarithm of volume
- logarithm of average closing price
- logarithmic average rate of return with its standard deviation, skewness and kurtosis

```python
crypto_scaled = scale(df)
crypto_scaled.std(axis=0)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1.])



Ultimately, all values used further are rescaled so that their mean is equal to 0 and their variance is equal to 1.

### 9.2. Dimension reduction using Principal Component Analysis (PCA)

The prepared set will be subjected to dimension reduction using the PCA method. Our goal is to retain as much of the variance of the original set as possible. The PCA algorithm itself is based on linear dimensionality reduction using singular value decomposition. It involves projecting the data into a space with fewer dimensions in order to best preserve the data structure. PCA analysis is based on determining the axis that preserves the highest variance value of the training set. The components are determined as linear combinations of the variables under study. The idea behind creating successive components is that they are not correlated with each other and are intended to maximise the variability that has not been explained by the previous component. Data standardisation is also used as a preliminary step here.


```python
pca = PCA(n_components=4)
crypto_pca1 = pca.fit_transform(crypto_scaled)
```

At the beginning, four PCA components will be used to examine how much information will be retained.

```python
df_crypto_pca1 = pd.DataFrame(data=crypto_pca1,
                             columns=["PC 1", "PC 2", "PC 3", "PC4"])
df_crypto_pca1.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
      <th>PC4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.172191</td>
      <td>-0.210540</td>
      <td>0.492125</td>
      <td>-1.177221</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.234535</td>
      <td>-0.263811</td>
      <td>0.522356</td>
      <td>1.048246</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208823</td>
      <td>-0.149446</td>
      <td>-0.350942</td>
      <td>0.011465</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.716515</td>
      <td>1.701024</td>
      <td>-1.506939</td>
      <td>-0.816227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.880847</td>
      <td>-0.605972</td>
      <td>-1.116188</td>
      <td>0.212009</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("With {} components, {:.2f}% of the initial variance of the set remains".format(4, pca.explained_variance_ratio_.sum()*100))
```

    With 4 components, 78.38% of the initial variance of the set remains.
    


```python
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_*100, color='green')
plt.title('Rozkład wariancji na komponenty')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_132_0.png)
    


In order to retain as much information as possible and as few components as possible, the PCA model will be adjusted to retain 90% of the variance.

```python
pca2 = PCA(n_components=.90)
crypto_pca = pca2.fit_transform(crypto_scaled)
```


```python
transformed_crypto_pca = pd.DataFrame(data=crypto_pca)
transformed_crypto_pca.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.172191</td>
      <td>-0.210540</td>
      <td>0.492125</td>
      <td>-1.177221</td>
      <td>0.607943</td>
      <td>0.261354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.234535</td>
      <td>-0.263811</td>
      <td>0.522356</td>
      <td>1.048246</td>
      <td>0.191015</td>
      <td>0.232522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208823</td>
      <td>-0.149446</td>
      <td>-0.350942</td>
      <td>0.011465</td>
      <td>-0.145460</td>
      <td>0.974179</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.716515</td>
      <td>1.701024</td>
      <td>-1.506939</td>
      <td>-0.816227</td>
      <td>0.104251</td>
      <td>-1.292999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.880847</td>
      <td>-0.605972</td>
      <td>-1.116188</td>
      <td>0.212009</td>
      <td>-1.403451</td>
      <td>-0.586729</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("With {} components, {:.2f}% of the initial variance of the set remains.".format(len(transformed_crypto_pca.columns), pca2.explained_variance_ratio_.sum()*100))
```

    With 6 components, 93.57% of the initial variance of the set remains.
    


```python
features = range(pca2.n_components_)
plt.bar(features, pca2.explained_variance_ratio_, color='green')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_137_0.png)
    


### 9.3. Clustering using the k-means method

The centroid algorithm involves finding the number of k-points that are the ‘centres of gravity’ of the centroids in the data set. The centre of gravity is the location representing the centre between all nearby points. The first centres of gravity are generated randomly and there must be as many of them as the number of clusters into which we want to divide our data set. In the next steps, we perform iterative calculations to optimise the positions of the centroids. The algorithm stops when the centroids (i.e. *i*) no longer change, i.e. they have stabilised, and when the defined number of iterations has been reached. It is worth noting that this type of approach does not detect concave clusters.

#### 9.3.1. Clustering the entire data set


```python
inertia = []
k = list(range(1, 16))
for i in k:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(crypto_scaled)
    inertia.append(km.inertia_)
```

The number of clusters (*n_clusters*) was set equal to the variable *i*. In the example above, inertia was used, which assumes that the resulting clusters consist of groups of convex points. Inertia is calculated as the distance between a given column (row, point) and the average value for the columns (rows). The greater the inertia, the further the points are from the average row/column profile. It does not work well with clusters that are more elongated and less diverse in shape.


```python
silhouttescore = []
ch_index = []
k = list(range(2, 16))
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42).fit(crypto_scaled)
    silhouttescore.append(metrics.silhouette_score(crypto_scaled, kmeans.labels_))
    ch_index.append(metrics.calinski_harabasz_score(crypto_scaled, kmeans.labels_))
```

The *silhouette_score* and *ch_index* lists were used to record the silhouette coefficient (average silhouette coefficient) and CH index results for all samples. In both cases, the highest value defines the most favourable number of clusters. The CH index was calculated using the following formula:
$$
CH = \frac{b}{a},
$$
 where:  
 a - average distance between points in a cluster to the centre of gravity of that cluster (cohesion),  
 b - average distance between the centres of gravity of the cluster and the global centre of gravity (separation).  

The Caliński-Harabasz index provides information about the similarity between an object and its own cluster in relation to other clusters. The higher the index value, the denser and better separated the clusters are. The calculated index value allows the most favourable number of clusters to be determined and enables clustering algorithms to be compared with each other.  
The silhouette value was calculated using the following formula:
$$
silhouette = \frac{(b-a)}{max(a,b)},
$$
 where:  
 a - average distance between cluster points,  
 b - average distance from the nearest cluster for each point.

From the above formula, it can be seen that if *a* is greater than *b*, then a particular observation is further away from observations in its group (average distance) and closer to observations in the neighbouring group. In this case, our equation takes the form:
$$
\frac{(b-a)}{a} = \frac{b}{a} - 1,
$$
Since *a* was greater, *b/a* will be less than 1, meaning that the entire equation will be less than 0. Similarly, if *a* is less than *b*, the value will be greater than 0.  
On the one hand, the value *a* informs us about the proximity of other elements within the group. This is not essentially density, but it provides information about how tightly packed the group is. On the other hand, the value of *b* shows how far the point is from the nearest other group. In other words, it indicates separation. The greater the value, the better the points are separated from each other. Therefore, it is best for *a* to be small (close to ‘allies’) and *b* to be large (far from ‘enemies’). Determining the maximum silhouette value can allow for the correct determination of the separation-concentration problem and thus enable the determination of the best *k*. It should be noted that the algorithm also informs whether a given grouping function has correctly grouped the values. If the silhouette coefficient is negative, it means that the values have been grouped incorrectly.
 


```python
fig, ax = plt.subplots(1,3,figsize=(18,5))
fig1 = sns.lineplot(x = range(1,16,1), y = inertia, ax=ax[0])
ax[0].set_title('Wykres bezwładności')
ax[0].set_xlabel('Liczba klastrów')
ax[0].set_ylabel('Bezwładność')
fig2 = sns.lineplot(x = range(2,16,1), y = silhouttescore, ax=ax[1])
ax[1].set_title('Wykres współczynnika sylwetki')
ax[1].set_xlabel('Liczba klastrów')
ax[1].set_ylabel('Silhoutte score')
fig3 = sns.lineplot(x = range(2,16,1), y = ch_index, ax=ax[2])
ax[2].set_title('Wykres indeksu CH')
ax[2].set_xlabel('Liczba klastrów')
ax[2].set_ylabel('Indeks Calińskiego-Harabasza')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_145_0.png)
    


#### 9.3.2. Clustering of the set reduced by PCA

In this approach, we use the PCA dimension reduction method, followed by the k-means algorithm. Therefore, the steps of the previous clustering are repeated:


```python
inertia_pca = []
k = list(range(1, 16))
for i in k:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(transformed_crypto_pca)
    inertia_pca.append(km.inertia_)
```


```python
silhouttescore_pca = []
ch_index_pca = []
k = list(range(2, 16))
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42).fit(transformed_crypto_pca)
    silhouttescore_pca.append(metrics.silhouette_score(transformed_crypto_pca, kmeans.labels_))
    ch_index_pca.append(metrics.calinski_harabasz_score(transformed_crypto_pca, kmeans.labels_))
```


```python
fig, ax = plt.subplots(1,3,figsize=(18,5))
fig1 = sns.lineplot(x = range(1,16,1), y = inertia_pca, ax=ax[0])
ax[0].set_title('Wykres bezwładności')
ax[0].set_xlabel('Liczba klastrów')
ax[0].set_ylabel('Bezwładność')
fig2 = sns.lineplot(x = range(2,16,1), y = silhouttescore_pca, ax=ax[1])
ax[1].set_title('Wykres współczynnika sylwetki')
ax[1].set_xlabel('Liczba klastrów')
ax[1].set_ylabel('Silhoutte score')
fig3 = sns.lineplot(x = range(2,16,1), y = ch_index_pca, ax=ax[2])
ax[2].set_title('Wykres indeksu CH')
ax[2].set_xlabel('Liczba klastrów')
ax[2].set_ylabel('Indeks Calińskiego-Harabasza')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_150_0.png)
    


#### 9.3.3. Choosing of the best model


```python
print("Wartości wskaźników dla modelu bez PCA:")
print("Silhoutte score: {:.3f}".format(max(silhouttescore)))
print("Indeks Calińskiego-Harabasza: {:.2f}".format(max(ch_index)))
print()
print("Wartości wskaźników dla modelu z PCA:")
print("Silhoutte score: {:.3f}".format(max(silhouttescore_pca)))
print("Indeks Calińskiego-Harabasza: {:.2f}".format(max(ch_index_pca)))
```

    Wartości wskaźników dla modelu bez PCA:
    Silhoutte score: 0.246
    Indeks Calińskiego-Harabasza: 31.52
    
    Wartości wskaźników dla modelu z PCA:
    Silhoutte score: 0.263
    Indeks Calińskiego-Harabasza: 36.76
    

The above metric values show that the model achieved better performance for data previously reduced by PCA. In both cases, the values are higher than those for the model without reduced dimensions. An important observation is the lack of consistency in the results of the model without PCA (graph from section 9.3.1). The Silhouette coefficient value indicates the optimal solution for 14 clusters, while the CH index value indicates 6 clusters. In the case of the model with dimensionality reduction using the PCA method (graph from subsection 9.3.2), the highest values of model goodness measures were calculated in both cases for 6 clusters, which means consistency of results. This information is another argument confirming the superiority of the model with dimensionality reduction over the model without it. Additionally, the inertia graph (section 9.3.2) shows the so-called ‘elbow’, i.e. the point where the inertia value decreases significantly more slowly. Only the model with 6 clusters implemented on data with reduced dimensionality was subjected to further analysis. 

### 9.4. Presentation of clustering results

#### 9.4.1. Final model results

The data will be clustered after passing through the PCA method, and then 6 clusters will be created, which will be presented in a histogram and a table assigning a given cryptocurrency to the appropriate cluster will be presented.


```python
clustered_pca = transformed_crypto_pca.copy()
model = KMeans(n_clusters=6, random_state=42)
model.fit(clustered_pca)
predictions = model.predict(clustered_pca)
```


```python
clustered_pca["name"] = df.index
clustered_pca["class"] = model.labels_
clustered_pca.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>name</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.172191</td>
      <td>-0.210540</td>
      <td>0.492125</td>
      <td>-1.177221</td>
      <td>0.607943</td>
      <td>0.261354</td>
      <td>1inchnetwork</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.234535</td>
      <td>-0.263811</td>
      <td>0.522356</td>
      <td>1.048246</td>
      <td>0.191015</td>
      <td>0.232522</td>
      <td>aave</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208823</td>
      <td>-0.149446</td>
      <td>-0.350942</td>
      <td>0.011465</td>
      <td>-0.145460</td>
      <td>0.974179</td>
      <td>algorand</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.716515</td>
      <td>1.701024</td>
      <td>-1.506939</td>
      <td>-0.816227</td>
      <td>0.104251</td>
      <td>-1.292999</td>
      <td>amp</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.880847</td>
      <td>-0.605972</td>
      <td>-1.116188</td>
      <td>0.212009</td>
      <td>-1.403451</td>
      <td>-0.586729</td>
      <td>ankr</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
clustered_pca_his = pd.DataFrame()
clustered_pca_his['class'] = clustered_pca['class'].value_counts()
clustered_pca_his
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
    </tr>
    <tr>
      <th>0</th>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(clustered_pca_his.index, clustered_pca_his['class'], color='green')
plt.title("Rozkład klas")
plt.xlabel('Nr klastra')
plt.ylabel('Liczba kryptowalut')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_160_0.png)
    


It can be seen that in the histogram above, only one currency has been assigned to each of the two clusters. These are a defunct cryptocurrency and a recently created cryptocurrency. Both tokens stand out significantly from the rest.


```python
Cluster0, Cluster1, Cluster2, Cluster3, Cluster4, Cluster5, = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
Cluster0['Klaster 0'] = clustered_pca['name'].loc[clustered_pca['class'] == 0]
Cluster1['Klaster 1'] = clustered_pca['name'].loc[clustered_pca['class'] == 1]
Cluster2['Klaster 2'] = clustered_pca['name'].loc[clustered_pca['class'] == 2]
Cluster3['Klaster 3'] = clustered_pca['name'].loc[clustered_pca['class'] == 3]
Cluster4['Klaster 4'] = clustered_pca['name'].loc[clustered_pca['class'] == 4]
Cluster5['Klaster 5'] = clustered_pca['name'].loc[clustered_pca['class'] == 5]
Cluster0.reset_index(drop=True, inplace=True)
Cluster1.reset_index(drop=True, inplace=True)
Cluster2.reset_index(drop=True, inplace=True)
Cluster3.reset_index(drop=True, inplace=True)
Cluster4.reset_index(drop=True, inplace=True)
Cluster5.reset_index(drop=True, inplace=True)
results = pd.concat( [Cluster0, Cluster1, Cluster2, Cluster3, Cluster4, Cluster5], axis=1)
results = results.fillna("")
results
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Klaster 0</th>
      <th>Klaster 1</th>
      <th>Klaster 2</th>
      <th>Klaster 3</th>
      <th>Klaster 4</th>
      <th>Klaster 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>avalanche</td>
      <td>1inchnetwork</td>
      <td>terraclassic</td>
      <td>ankr</td>
      <td>aave</td>
      <td>aptos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>axieinfinity</td>
      <td>algorand</td>
      <td></td>
      <td>chiliz</td>
      <td>arweave</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>binancecoin</td>
      <td>amp</td>
      <td></td>
      <td>decentraland</td>
      <td>balancer</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>bitcoin</td>
      <td>aragon</td>
      <td></td>
      <td>dogecoin</td>
      <td>binaryx</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>cardano</td>
      <td>astar</td>
      <td></td>
      <td>ecash</td>
      <td>compound</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>chainlink</td>
      <td>audius</td>
      <td></td>
      <td>fantom</td>
      <td>dash</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>cosmos</td>
      <td>celo</td>
      <td></td>
      <td>gala</td>
      <td>decred</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>eos</td>
      <td>convexfinance</td>
      <td></td>
      <td>holo</td>
      <td>flux</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>ethereum</td>
      <td>flow</td>
      <td></td>
      <td>iotex</td>
      <td>gmx</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>litecoin</td>
      <td>harmony</td>
      <td></td>
      <td>oasisnetwork</td>
      <td>gnosis</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>monero</td>
      <td>hive</td>
      <td></td>
      <td>polygon</td>
      <td>hedera</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>neo</td>
      <td>icon</td>
      <td></td>
      <td>ravencoin</td>
      <td>helium</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>pancakeswap</td>
      <td>immutablex</td>
      <td></td>
      <td>reserverights</td>
      <td>horizen</td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>polkadot</td>
      <td>just</td>
      <td></td>
      <td>ripple</td>
      <td>injective</td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>solana</td>
      <td>kava</td>
      <td></td>
      <td>siacoin</td>
      <td>internetcomputer</td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>uniswap</td>
      <td>loopring</td>
      <td></td>
      <td>sxp</td>
      <td>kadena</td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td></td>
      <td>masknetwork</td>
      <td></td>
      <td>thegraph</td>
      <td>klaytn</td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td></td>
      <td>mina</td>
      <td></td>
      <td>thetanetwork</td>
      <td>kusama</td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td></td>
      <td>naxo</td>
      <td></td>
      <td>vechain</td>
      <td>lisk</td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td>nem</td>
      <td></td>
      <td>winklink</td>
      <td>livepeer</td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td>optimism</td>
      <td></td>
      <td>zilliqa</td>
      <td>maker</td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td></td>
      <td>osmosis</td>
      <td></td>
      <td></td>
      <td>moonbeam</td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td></td>
      <td>ox</td>
      <td></td>
      <td></td>
      <td>omgnetwork</td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td></td>
      <td>qtum</td>
      <td></td>
      <td></td>
      <td>quant</td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td></td>
      <td>smoothlovepotion</td>
      <td></td>
      <td></td>
      <td>sushiswap</td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td></td>
      <td>stacks</td>
      <td></td>
      <td></td>
      <td>terra</td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td></td>
      <td>stellar</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td></td>
      <td>stepn</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td></td>
      <td>storj</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td></td>
      <td>synthetix</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td>tezos</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td></td>
      <td>threshold</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td></td>
      <td>waves</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td></td>
      <td>wax</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td></td>
      <td>woonetwork</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Cryptocurrencies with the highest exchange rates, market capitalisation and rates of return, such as Bitcoin and Ethereum, were assigned to cluster 0. This means that the model effectively selected market leaders for one group, as it included thriving cryptocurrencies such as LiteCoin, Cardano, Polkadot, Solana and BinanceCoin. A major advantage of the model is that it separates currencies with very low exchange rates that have collapsed, such as terraclassic. This gives a clear indication that they should be avoided.

#### 9.4.2. Visualisation of results in 2D


```python
tsne = TSNE()
tsne_features = tsne.fit_transform (transformed_crypto_pca)
```

The data is reduced to two dimensions using the t-SNE method. This is a non-linear dimensionality reduction technique based on the similarity between points. This method focuses on preserving the local structure of the data and works by minimising the distance between points in a Gaussian distribution.


```python
k_means_df = pd.DataFrame(tsne_features).reset_index(drop=True)
k_means_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-7.104079</td>
      <td>3.154223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.052576</td>
      <td>0.785348</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.849134</td>
      <td>1.199553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-5.888927</td>
      <td>-0.670073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-11.895643</td>
      <td>-0.036419</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = KMeans(n_clusters=6, random_state=42)
model.fit(k_means_df)
predictions = model.predict(k_means_df)
k_means_df["class"] = model.labels_
k_means_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-7.104079</td>
      <td>3.154223</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.052576</td>
      <td>0.785348</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.849134</td>
      <td>1.199553</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-5.888927</td>
      <td>-0.670073</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-11.895643</td>
      <td>-0.036419</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The two-dimensional set of cryptocurrencies has been divided into 6 classes.


```python
k_means_his = pd.DataFrame()
k_means_his['class'] = k_means_df['class'].value_counts()
k_means_his
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
    </tr>
    <tr>
      <th>0</th>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(k_means_his.index, k_means_his['class'], color='green')
plt.title("Rozkład klas")
plt.xlabel('Nr klastra')
plt.ylabel('Liczba kryptowalut')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_171_0.png)
    



```python
plt.scatter(k_means_df[0], k_means_df[1], c=k_means_df['class'])
plt.title("Wizualizcja podziału kryptowalut na przestrzeni 2D")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_172_0.png)
    



```python
plt.figure(figsize = (14,10))
plt.scatter(k_means_df[0], k_means_df[1], c=k_means_df['class'])
plt.title("Wizualizcja podziału kryptowalut na przestrzeni 2D (z podpisami)")
plt.xlabel('PC1')
plt.ylabel('PC2')
for i, s in enumerate(df.index):
    plt.annotate(s, (k_means_df[0][i], k_means_df[1][i]), fontsize=9)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_173_0.png)
    


Clustering two-dimensional data is purely visual and should not be used to guide investment decisions. However, some information has been retained, allowing us to observe that the largest cryptocurrencies are located in two clusters (yellow and green in the chart). This time, the model was unable to identify cryptocurrencies that clearly stand out from the rest, which undermines its credibility.



