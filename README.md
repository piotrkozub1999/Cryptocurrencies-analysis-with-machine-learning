<h1><center>Cryptocurrencies Market Analysis</center></h1>
<h2><center>Machine Learning</center></h2>
<h3><center>PL</center></h3>

## 1. Cel i założenia projektowe

Celem projektu jest przeprowadzenie badań dotyczących zachowania rynku kryptowalut. Projekt zawiera analizę zachowania kursów najpopularniejszych kryptowalut na rynku światowym. W tym celu przeprowadzono ogólne rozpoznanie rynku kryptowalut polegające na sprawdzeniu kapitalizacji rynkowej czy udziału stu najpopularniejszych kryptowalut na rynku. Wykonano również analizę Bitcoina jako głównej kryptowaluty, która posiada znaczny udział w rynku kryptowalut i oddziałuje na kursy innych aktywów cyfrowych. Sprawdzono zachowanie Bitcoina w odniesieniu do najważniejszych wydarzeń na świecie oraz innych ciekawych czynników. Zestawiono również dane różynch kryptowalut w celu zweryfikowania zwiazków między nimi. Projekt obejmuje także analizę w odniesieniu do aktywów rzeczywistych na giełdzie takich jak indeks S&P 500, indeks Dolara czy kurs złota i srebra. W analizie zostały również wykorzystane techniki eksploracji danych w celu pogrupowania kryptowalut pod względem ryzyka inwestowania.

## 2. Dane projektowe

Dane wykorzystane do analizy zostały pobrane z giełdy Binance, która jest największą giełdą kryptowalut. Gromadzi ona dane od momentu pojawienia się danej kryptowaluty na giełdzie. Oznacza to kompletność danych poszczególnych kryptowalut w zależności od czasu pojawienia się ich na giełdzie Binance. Podczas analizy wykorzystano odpowiednie okresy czasowe tak by istniała możliwość pobrania kompletnego zbioru danych wszystkich potrzebnych kryptowalut.  

Dostęp do kursów surowców uzyskano ze strony data.nasdaq natomiast notowania indeksu S&P 500 i indeksu dolara pobrano ze strony Yahoo Finance. Giełdy aktywów rzeczywistych w odróżnieniu od giełd kryptowalut pracują jedynie w dni robocze, co oznacza brak danych w dni wolne od pracy. Rozwiązaniem tego problemu jest odpowiednie przetworzenie danych kryptowalut by odpowiadały datom pobranych notowań giełdowych. 

## 3. Wykorzystane narzędzia

Analiza została wykonana przy użyciu języka programowania Python wraz z gotowymi paczkami umożliwiającymi pobieranie, przetwrzanie oraz wizualizację danych:
* Biblioteka *Binance* umożliwia pobranie danych zgromadzonych w serwisie Binance, niezbędnych do analizy kursów kryptowalut.
* Biblioteki *quandl* oraz *yfinance* służą do pobierania danych aktywów z giełdy z stron związanych z ekonomią i finansami.
* Biblioteki *Pandas*, *NumPy* oraz *SciPy* umożliwiają odpowiednie przetwarzanie danych pobranych z API. Dodatkowo biblioteka *datetime* pozwala na konwersję danych czasowych.
* Biblioteki *Matplotlib* oraz *seaborn* są bibliotekami, które generują wykresy na podstawie danych. Specjalnie na potrzeby analizy finansowej na podstawie wykresów kursów kryptowalut wykorzystano również bibliotekę *mplfinance*. 
* Biblioteka *scikit-learn* jest biblioteką wprowadzającą funkcję uczenia maszynowego oraz zawiera różne metody klasyfikacji, regresji oraz klastrowania.
* Biblioteka *statsmodels* ułatwia eksploracje danych oraz szacowanie modelów statystycznych.
* Biblioteka *Warnings* służy do wykrywania błędów i informowaniu o ostrzeżeniach.


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

## 4. Pobieranie danych z giełdy Binance

### 4.1. Dostęp do giełdy Binance


```python
apikey = '...'
secret = '...'
client = Client(apikey, secret)
```

 Dostęp do danych z API serwisu Binance użyto kodów pełniących funkcję kluczy dostępu (API key oraz secret key). API key, jest to klucz przypisany do zmiennej *apikey* oraz secret key, który przypisujemy do zmiennej *secret*. Oba te klucze są niezbędne do uzyskania danych z API giełdy Binance oraz korzystania wraz z przetwarzaniem danych w dalszej części projektu. Samo uzyskanie kluczy dostępu nie jest rzeczą powszechnie dostępną, gdyż oprócz konta na serwisie Binance, które trzeba weryfikować dowodem osobistym, należy również posiadać min. 90 zł zainwestowanych w kryptowalutach. Po spełnieniu tych kryteriów istnieje możliwość uzyskania dostępu do kluczy dostępu.

Za pomocą modułu *Client* biblioteki *Binance* uzyskano pełen dostęp do danych giełdy Binance.

### 4.2. Pobieranie danych kryptowalut


```python
def data_change(data):
    data_pd = pd.to_datetime(data, format='%d %m %Y')
    data_int = pd.to_datetime(data_pd, unit='s').value
    return int(data_int/1000000)
```

Funckja *data_change* jest odpowiedzialna za konwertowanie daty zapisanej w formie "dzień miesiąc rok" na ilość milisekund, które mineły od 1 stycznia 1970 roku.


```python
interval = Client.KLINE_INTERVAL_1WEEK
begin = '01 01 2019'
end = '30 11 2022'
begin_int = data_change(begin)
end_int = data_change(end)
```

Wprowadzono zmienne odpowiadające za pobieranie danych z giełdy. Zmienna *interval* definiuje częstotliowść uzyskania danych z giełdy. Natomiast zmienne *begin* i *end* definiują datę początową i końcową pobierania danych. Daty następnie są zamieniane na ilość milisekund, które mineły od 1 stycznia 1970 roku.

W projekcie wykorzystujemy dane od początku roku 2019 do teraz. Pobieramy je z tygodniowym interwałem.


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

Funkcja *take_coin* służy do pobrania danych danej kryptowaluty względem USDT (cyfrowy odpowiednik dolara) na podstawie nazwy i symbolu. Dodano również nowe kolumny do pobranych danych:
* 'slug' - unikalny symbol każdej kryptowaluty, wprowadzony by naprawić zduplikowane tokeny udostępniane za pomocą symbolu lub nazwy,
* 'symbol' - symbol danej kryptowaluty (ticker),
* 'name' - nazwa kryptowaluty,
* 'ranknow' - miejsce w rankingu najpopularniejszych kryptowalut


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

Pobranie danych z 100 najpopularniejszych kryptowalut na podstawie informacji ze strony CoinMarketCap. Pomijamy stable coiny symulujące cenę dolara do którego się odnosimy.

### 4.3. Przetwarzanie danych


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
      <th>Czas otwarcia</th>
      <th>Kurs na otwarciu</th>
      <th>Najwyższy kurs w trakcie dnia</th>
      <th>Najniższy kurs w trakcie dnia</th>
      <th>Kurs na zamknięciu</th>
      <th>Ilość kryptowaluty w obiegu</th>
      <th>Czas zamknięcia</th>
      <th>Zmiana wolumenu transakcji kryptowaluty</th>
      <th>Liczba transakcji</th>
      <th>Ilość wolumenu zrealizowanego</th>
      <th>Ilość wolumenu niezrealizowanego</th>
      <th>Slug</th>
      <th>Symbol</th>
      <th>Nazwa</th>
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



Podzielenie pobranych danych na odpowiednie kolumny. Natępuje również zamiana na wartości numerycznych zapisanych za pomocą typu *string* na wartości typu *float*, w celu poprawnej analizy danych na wykresach. Wartości czasowe w postaci milisekund zamieniane są na odpowiadjące daty. Usuwamy niepotrzebną kolumnę '*Wartość do zignorowania*'.

Bazując na pobranych danych tworzymy nowy *DataFrame* "crypto" w celu zachowania kolumn potrzebnych do dalszej analizy oraz poprawy estetyki.
W tym punkcie tworzymy kilka nowych zmiennych:
- *market* - wartość kapitalizacji danej kryptowaluty
- *close_ratio* - miara straty wartości na koniec każdej sesji (stosunek ceny maksymalnej do ceny zamknięcia)
- *spread* - amplituda wartości w trakcie jednej sesji

### 4.4 Wprowadzenie danych logarytmicznych

W celu przeanalizowania dynamiki kryptowalut w czasie zmodyfikowano dane poprzez:
* wprowadzenie logarytmicznej skali w okresie lat 2019-2022 z powodu dużego (wykładniczego) tempa zmian wartości kryptowalut (w porównaniu do klasycznej giełdy),
* wstawienie wartości '0' w miejsce ewentualnie brakujących danych,
* usunięcie błędnie pobranych wierszy. Nieliczne kryptowaluty posiadały dodatkowe wiersze, w tym celu zostawiamy jedynie rzędy w których data zgadza się z datami bitcoina
* dodanie zmiennej *birth_time* informujący o wieku (niektóre krytpowaluty powstały po 2019 roku)
* dodanie zmiennej *log_return* informujące o stopie zwrotu inwestora tydzień do tygodnia


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



## 5. Przegląd rynku kryptowalut


```python
crypto_market = crypto.groupby('date')[['market']].sum()
```

Grupowanie i zsumowanie danych po dacie dla kolumny '*market*' oraz '*volume*', w celu uzyskania informacji o rozmiarze rynku oraz wolumenie kryptowalut w danym dniu.


```python
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title('Całkowita wartość rynku')
plt.xlabel('Data')
plt.ylabel('Wartość rynku')
sns.lineplot(data=crypto_market.market, color="#0000ac", label='Kapitalizacja rynkowa')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_37_0.png)
    


No powyższym wykresie możemy zauważyć, że wartość rynku kryptowalut była najwyższa w połowie 2021 roku. Po bardzo szybkim wzroście nastąpiło równie szybkie załamanie rynku.

W celu pokazania aktualnego rozkładu kapitalizacji rynkowej na giełdzie Binance, porównamy jej wartości dla ostatniego dnia naszych szeregów czasowych, tj. 12.04.2022.
Porównanie zostanie zwizualizowane na wykresie kołowym dla 10 największych kryptowalut.


```python
crypto_snap['composition'] = np.where(crypto_snap.ranknow <= 10, crypto_snap.ranknow, 11)
crypto_market_comp = crypto_snap.groupby(by = ['composition'])['market'].sum()
```

Stworzenie jedynastu przedziałów, które bedą zawierać dziesięć najpopularniejszych kryptowalut oraz wartość zsumowaną od jedynastej do setnej kryptowaluty.


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
    


Jak możemy zauważyć, zdecydowana większość wartości rynku należy do Bitcoina. O ponad połowę mniej warte jest Ethereum, zaś cała reszta zamyka się w 25% kapitalizacji.


```python
crypto_4 = crypto[(crypto.ranknow <= 4)]
```

## 6. Analiza najpopularniejszych kryptowalut

Analizując najbardziej popularne kryptowaluty od początku 2019 roku największą stabliność wykazuje Bitcoin oraz Ethereum, które w danym przedziale czasowym zawsze zajmowały odpowiednio miejsce 1 i 2. Przez większość czasu, bo aż do początku 2021 roku 3 miejsce cały czas stabilnie należało do Ripple ale wzrost popularności giełdy Binance spowodował, że należąca do nich kryptowaluta zyskała na wartości i od 2021 roku stale jest trzecią najpopularniejszą kryptowalutą. W tym czasie Ripple osylował w top6 pupularnych kryptowalut.  
Można więc z tego wywnioskować, że na przestrzeni ostatnich 4 lat najbardziej stabilnymi kryptowalutami były Bitcoin, Ethereum, BinanceCoin oraz Ripple.



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
    


Powyższy wykres przedstawia wartość rynkową TOP4 kryptowalut na giełdzie binance. Skala logarytmiczna nie obrazuje jednak rzeczywistej przewagi Bitcoina nad resztą.
Z drugiej strony niezastosowanie skali logarytmicznej sprawi że przebieg wartości pozostałych kryptowalut przybliży się do osi x.


```python
sns.pairplot(crypto_4[['name','log_market','log_volume','log_return','log_close','spread','close_ratio']], hue='name')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_51_0.png)
    


Powyższy zbiór wykresów przedstawie zależności pomiędzy pięcioma wartościami (wartość rynku, wolumen zrealizowany, stopa zwrotu, wspołczynnik zamknięcia, tygodniowa amplituda i wartość na zamknięciu dla czterech największych kryptowalut.
Obserwując poszczególne zależności możemy zauważyć, że kombinacje wykresów związane z wolumenem zrealizowanym oraz wartością są rozdzielne. W większości przypadków zbiory te mozna łatwo odseparować dla poszczególnych kryptowalut. Pozostałe zmienne z reguły nachodzą na siebie.


```python
fig, ax = plt.subplots(figsize= (12,8))
sns.boxplot(x='name', y='log_return', data = crypto[crypto.ranknow <= 10], ax = ax)
ax.set_title('Spektrum wydajności dla 10 najpopularniejszych kryptowalut')
ax.set_xlabel('')
ax.set_ylabel('Wartość logartymiczna stopy zwrotu')
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_53_0.png)
    


Powyższy wykres pudełkowy przedstawia rozkład stopy zwrotu dla poszczegolnych kryptowalut.
Najbardziej nasuwający się wniosek wynika z rozmieszczenia wartości odstających. Możemy zaobserwować, że większość kryptowalut, w przeciwieństwie do Bitcoina, posiada pojedyńcze przypadki zdecydowanych spadków tydzień do tygodnia. Oznacza to, że ich zakup niesie ze sobą niskie, ale możliwe ryzyko gwałtownego spadku wartości.

## 7. Analiza Bitcoina


```python
bitcoin = crypto[crypto.slug == 'bitcoin']
bitcoin['date'] = pd.to_datetime(bitcoin['date'])
```

Stworzenie zmiennej odpowiedzialnej za wszystkie dane dotyczące tylko Bitcoina


```python
mpf.plot(bitcoin.set_index('date').tail(len(bitcoin)),
        figscale=1.5, type='candle', style='yahoo',
        volume=True, title='Wykres Bitcoina do USDT', mav=(5))
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_58_0.png)
    


Powyższy wykres przedstawia kurs Bitcoina względem krypotwaluty Tether, która jest cyfrowym odpowiednikiem dolara i oscyluje wokół wartości jednego dolara za token.  

Od 2019 roku duże amerykańskie korporacje zaczęły akceptować nową metodę płatności za pomocą waluty cyfrowej w postaci Bitcoina. Spowodowało to wzrost notowań do 10000$ a następnie oscylację wokół wartości 8500$. Pod koniec roku 2019 wybuchła pandemia Covid w Azji co przełożyło się na kolejny wzrost notowań kryptowaluty. Powodem był początek kryzysu w krajach przeciwstawiających się kryptowalutom jako formie płatności (np. Chiny). W momencie gdy pandemia dotarła do Europy i USA kurs Bitcoina mocno spadł i a tokeny zostały w dużym stopniu wyprzedawane. Pod wpływem niestabilności gospodarki wartość kryptowaluty zaczęła znów wzrastać wyrównując wartość osiągniętą przed pandemią (około 1000$) i stabilizując osiągniętą wartość przez ponad pół roku.  

Pod koniec 2020 roku zaczął się największy w historii Bitcoina wzrost kursu, który sięgnął wartości ponad 61000$. Wartość zmieniła się o ponad 500% w 4 miesiące. Przyczyną dużego wzrostu kursów kryptowalut były skutki kryzysu powstałego przez pandemię. Zaczęto poszukiwać rozwiązania trudnej sytuacji gospodarczej oraz coraz większe firmy tj. Amazon, Bank of New York czy Morgan Stanley pozwalały na dokonywanie transakcji za pomocą Bitcoina. Osoby wpływowe np. Elon Musk zaczęły inwestować w tokeny BTC i rozpowszechniać informacje na ich temat, co również spowodowało wzrost kursu. W pewnym momencie po ustabilizowaniu najwyższej wartości kryptowaluty nastąpiła ogólna panika i nagłe masowe wyprzedanie kryptowaluty przez krótkoterminowych inwestorów. Rezultatem był nagły spadek kursu o ponad 40%. Nowa wartość BTC wynosiła około 33000$. Po masowej panice przyszedł jak zawsze okres wzrostu wartości, który został zatrzymany przez rząd Chin. We wrześniu 2021 roku Ludowy Bank Chin wprowadził zakaz transakcji kryptowalut co odbiło się na notowaniach giełdowych Bitcoina. Wzrost został zahamowany, lecz tylko na krótki okres po, którym wartość Bitcoina sięgnęła rekordowych ponad 64000$ za 1 BTC. Jest to największa wartość Bitcoina w swojej historii. 

Pod koniec 2021 roku rozpoczął się spadek kursu Bitcoina spowodowany korektą notowań. Po długiej hossie jaka wystąpiła w 2021 roku przyszedł czas na bessę. Spadek kurs trwał aż do lutego bieżącego roku kiedy to wartość kryptowaluty osiągnęła duże oscylacje wokół 40000$. Pod wpływem wybuchu wojny między Rosją i Ukrainą wartość Bitcoina pierw wzrosła ale w związku z dużą inflacją oraz niepewnością gospodarczą kurs zaczął ponowie spadać. Kolejnym czynnikiem wpływającym na kurs było coraz mniejsza ilość tokenów możliwych do wydobycia, a co za tym idzie coraz trudniejszy proces. Od czerwca kurs Bitcoina ustabilizował się i oscyluje wokół 20000$ z lekką tendencją spadkową.  

Token BTC jest bardzo związany z aktualną sytuacją polityczną i gospodarczą. Na kurs
kryptowaluty mają duży wpływ zarówno kraje i ich decyzje jak i duże liczące się na giełdzie
spółki. Wartość Bitcoina może być w łatwy sposób manipulowana w zależności od decyzji
podjętych przez osoby wysoko postawionych co ma ujemny wpływ na niezależność
kryptowaluty.



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



Wyświetlenie statystyk dla każdej kolumny danych Bitcoina:
* 'count' - liczba wierszy, suma długości wszystkich szeregów czasowych
* 'mean' - wartość średnia
* 'std' - wartość odchylenia standardowego
* 'min' - wartość minimalna
* '25%' - kwartyl dolny, percentyl dwudziesty piąty
* '50%' - wartość środkowa, mediana, percentyl pięćdziesiąty
* '75%' - kwartyl górny, percentyl siedemdziesiąty piąty
* 'max' - wartość maksymalna

Powyższa tabela ze statystykami pozwala na dokładniejszą analizę kursu Bitcoina. Wartość minimalna jaką osiągnął Bitcoin od początku 2019 roku wynosi 3349,92$ natomiast największa 69000$. Odchylenie standardowe jest bardzo duże co oznacza duże zróżnicowanie notowań kryptowaluty. Wartość percentyli powierdza, że przez większość czasu wartości osiągały jednak niższe notowania. Wartość średnia wynosi około 23700$ natomiast mediana niecałe 18300$. Występuje dość duża różnica między tymi wartościami co powtierdza duże zróżnicowanie kursu oraz krótkie okresy czasu gdy wartość Bitcoina osiągał największe wartości.


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

Funkcja *plot_time_series* służy do tworzenia wykresu rozkładu szeregów czasowych danej kryptowaluty oraz wykresu prawdpodobieństwa, w celu analizy poszczególnych kryptowalut.


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



Wyliczone statyski stopy zwrotu Bitcoina pokazują, że odchylenie standardowe jest wysokie w stosunku do wartości średniej.
Skośność równa -0.7 oznacza ogon lewostronny histogramu, co możemy zaobserwować na powyższym wykresie rozkładu stopy zwrotu.
Dodatnia kurtoza mówi o większej ilości dodatnich wartości odstających, co również jest pokazane na wykresie (wyższe słupki z dodatnimi wartościami).
Obie te statystki świadczą o tym, że rozkład stopy zwrotu jest daleki od rozkładu normalnego.


## 8. Analiza kryptowalut względem aktywów na giełdzie

Założeniem tej części analizy jest porównie aktywów cyfrowych z aktywami rzeczywistaymi na giełdach, w celu znalezienia korleacji pomiędzy poszczególnymi aktywami. Spośród wcześniej prezentowanych stu najpopularniejszych kryptowalut wybrano Bitcoina, Ethereum, Binance Coin oraz Ripple. W przypadku aktywów giełdowych wybrano kurs surowców takich jak złoto i srebro. Wybrano również kurs indeksu S&P 500 oraz indeksu dolara czyli stosunek dolara amerykańskiego do koszyka walut obcych.

### 8.1. Pobranie danych z giełdy kryptowalut


```python
interval = Client.KLINE_INTERVAL_1DAY
begin_int = data_change('01 01 2022')
end_int = data_change('10 12 2022')
```

W celu wykonania dokładniejszej analizy postanowiono zmienić daty na okres od początku do końca 2022 roku. Zmianie uległa również częstotliwość pobierania danych. Dane są aktualizowane dziennie a nie tygodniowo jak było to wykonane podczas pierwszej analizy.


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


Stworzenie nowej tabeli z przetworzonymi danymi spersonalizowanymi, które zostaną wykorzystane podczas analizy.


```python
crypto2022['log_close'] = np.log(crypto2022.close)
crypto2022['log_volume'] = np.log(crypto2022.volume)
crypto2022['log_market'] = np.log(crypto2022.market)
crypto2022['spread'] = (crypto2022.high - crypto2022.low) / crypto2022.close
crypto2022['log_return'] = np.log(crypto2022.close / crypto2022.close.shift(1))
crypto2022 = crypto2022.replace([np.inf,-np.inf, np.nan], 0)
```

Wprowadzenie wartości logarytmicznych identycznie jak podczas wcześniejszej analizy. Wykorzystano również zamianę wartości brakujących na wartość równą 0.


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

Stworzenie tabeli zawierającej datę oraz wartość logarytmiczną zwrotu dla wcześniej podanych kryptowalut. Dane zawarte w tabeli są odpowiednio przetworzone.


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



### 8.2. Pobieranie danych aktywów rzeczywistych z giełdy


```python
quandl.ApiConfig.api_key = 'jou3Hy9N_sKPZxy9mgxt'
```

Dostęp do danych zawartych na stronie data.nasdaq (dawniej quandl) uzyskujemy poprzez specjalny klucz dostępu (*api_key*).


```python
start = datetime(2022, 1, 1)
end = datetime(2022, 12, 10)
```

Wprowadzenie przedziału czasowowego zgodnego z poprzednim podpunktem, w któym pobierano dane dla trzech kryptowalut również od początku roku 2022.


```python
gold_price = quandl.get("LBMA/GOLD", start_date = start, end_date = end)
silver_price = quandl.get("LBMA/SILVER", start_date = start, end_date = end)
stock_index = yf.download("^GSPC", start, end)
USD_index = yf.download('DX-Y.NYB', start, end)
```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    

Dane notowań złota i srebra względem różnych walut w zdefiniowanych wcześniej ramach czasowych pobrano z data.nasdaq. Pobrano również notowania indeksu S&P 500 oraz indeksu dolara ze strony Yahoo Finance.


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

Połączenie w tabelę wcześniej pobranych danych aktyw giełdowych. Pozostawiona zostaje jedyna cena na zamknięciu każdego dnia.


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



Kolejno wszystkie ceny zostają zamienione na logarytmiczną stopę zwrotu.


```python
for col in stock_market_asset.columns:
    stock_market_asset[col] = np.log(stock_market_asset[col] / stock_market_asset[col].shift(1))
stock_market_asset.head(len(stock_market_asset))
stock_market_asset.drop(index=stock_market_asset.index[0], axis=0, inplace=True)
```


Pozostawione zostały jedynie wspólne daty, tak by liczba wierszy się zgadzała. Taki zabieg jest niezbędny ze względu na prace giełdy jedynie w dni robocze, podczas gdy giełda kryptowalut jest aktywna codziennie.


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



Połączenie tabeli z danymi kryptowalut oraz notowań giełdowych, w celu analizy korelacji między tymi notowaniami.

### 8.3. Analiza korelacji


```python
corr_matrix = eight_assets.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_110_0.png)
    


Powyższy wykres prezentuje macierz korelacji  stopy zwrotu opisanych wyżej aktywów. Można zauważyć, że korelacja pomiędzy wszystkimi kryptowalutami jest bardzo silna i wynosi co najmniej 0,76 a maksymalnie 0,9 (pomijając korelację między tymi samymi aktywami).  
W przypadku surowców występują takie same zależności co między kryptowalutami co oznacza, że kurs złota i srebra są mocno ze sobą skorelowane. Jeśli sprawdzimy wpływ kryptowalut na owe surowce korelacja jest bliska wartości 0 co oznacza, że nie występuje praktycznie żadna korelacja między nimi.  
Trzecim aktywem rzeczywistym są notowania S&P 500, które wykazują korelację w stopniu umiarkowanym z kryptowalutami. Największą korelację mają zarówno Bitcoin oraz Ethereum jako kryptowaluty najbardziej wpływowe na rynku. Powiązanie notowań indeksu S&P 500 z wartością złota i srebra jest znikome.
Ostatnim analizowanym aktywem jest indeks dolara  W wszystkich przypadkach indeks dolara jest skorelowany z innymi aktywami jako główna waluta transakcji na świecie. Co ważne korelacja we wszystkich przypadkach jest ujemna czyli wzrost wartości dolara powoduje obniżenie wartości notowań pozostałych aktywów. Korelacja dolara z kryptowalutami oraz surowcami jest słaba i wynosi odpowiednio około -0,3 i -0,25. Korelacja względem indeksu S&P 500 wynosi -0,48, co potwierdza poprawność wykonania przetwarzania danych.

## 9. Zastosowanie wybranych algorytmów uczenia maszynowego

W tym rozdziale stworzony zostanie model bazowy służący do pogrupowania wszystkich analizowanych kryptowalut. Po odpowiednim przygotowaniu danych wykonane zostanie:
- zmniejszenie wymiarowości za pomocą metody PCA
- klastrowanie metodą k-średnich
- ocena modeli za pomocą współczynników: Silhoutte Score oraz indeks Calińskiego-Harabasza
- przedstawienie wyników klasteryzacji

Oczekiwanym rezultatem klasteryzacji jest pogrupowanie kryptowalut, dzięki czemu możliwa będzie ocena, w które z nich warto zainwestować, a których lepiej unikać.

### 9.1. Przygotowanie danych


```python
crypto_mean = pd.DataFrame(crypto.groupby('slug').mean())
```

Dane wykorzystywane w klastrowaniu nie mogą przyjmować formy szeregów czasowych. Od teraz każda kryptowaluta będzie posiadać jedną wartość w każdej kolumnie. W tym celu obliczona została średnia wartość dla wszystkich kompentów danej kryptowaluty.


```python
crypto_log_return = crypto.groupby('slug').log_return
ret_std = pd.DataFrame(crypto_log_return.std())
ret_skew = pd.DataFrame(crypto_log_return.skew())
ret_kurt = pd.DataFrame(crypto_log_return.apply(pd.DataFrame.kurt))

ret_std.columns = ['std']
ret_skew.columns = ['skew']
ret_kurt.columns = ['kurt']
```

Ponadto obliczone zostały dodatkowe wskaźniki bazujące na średniej z logarytmicznej stopy zwrotu każdej kryptowaluty:
- odchylenie standardowe
- skośność
- kurtoza


```python
df = pd.merge(crypto_mean, ret_std, on = 'slug')
df = pd.merge(df, ret_skew, on = 'slug')
df = pd.merge(df, ret_kurt, on = 'slug')
df.dropna(inplace = True)
```


Wybranie danych do klastrowania:


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



Model klastrujący nie może zawierać naszych wszystkich kolumn, gdyż niektóre z nich nie mają żadnego wpływu na przynależność kryptowaluty pod kątem opłacalności. Badania danych takich jak ceny w ciągu dnia lub ranking krytpowaluty niepotrzebnie zakłamałyby otrzymane wyniki.
W tym celu pozostawione zostaną jedynie następujące parametry:
- stosunke ceny zamknięcia do otwarcia
- spread
- logarytm kapitalizacji rynkowej
- logorytm wolumenu
- logartm średniej ceny zamknięcia
- średnia logarytmiczna stopa zwrotu wraz z jej odchyleniem standardowym, skośnością i kurtozą


```python
crypto_scaled = scale(df)
crypto_scaled.std(axis=0)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1.])



Ostatecznie wszystkie wykorzystywane dalej wartości zostają przeskalowne, tak by ich średnia była równa 0, a wariancja 1.

### 9.2. Redukcja wymiarowości za pomocą Analizy Składowych Głównych (PCA)

Przygotowany zbiór poddany zostanie redukcji wymiarowości metodą PCA. Naszym celem jest pozostawienie jak największej ilości wariancji oryginalnego zbioru. Sam algorytm PCA bazuje na liniowej redukcji wymiarowości przy użyciu rozkładu wartości osobliwych. Polega on na rzutowaniu danych do przestrzeni o mniejszej liczbie wymiarów tak, aby jak najlepiej zachować strukturę danych. Analiza PCA opiera się o wyznaczanie osi zachowującej największą wartość wariancji zbioru uczącego. Składowe wyznaczamy jako kombinacje liniową badanych zmiennych. Idea tworzenia kolejnych składowych polega na tym, że kolejne składowe nie są skorelowane ze sobą oraz mają na celu zmaksymalizować zmienność, która nie została wyjaśniona przez poprzednią składową. Jako wstępny etap jest tu również wykorzystywana standaryzacja danych.


```python
pca = PCA(n_components=4)
crypto_pca1 = pca.fit_transform(crypto_scaled)
```

Na początku wykorzystane będą 4 składowe komponenty PCA, w celu zbadania ile informacji zostanie zachowanych.


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
print("Przy {} komponentach składowych pozostało {:.2f}% początkowej wariancji zbioru".format(4, pca.explained_variance_ratio_.sum()*100))
```

    Przy 4 komponentach składowych pozostało 78.38% początkowej wariancji zbioru
    


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
    


W celu zachowania jak największej ilości informacji i jak najmniejszej ilości komponentów model PCA zostanie dostosowany tak, by zachował 90% wariancji.


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
print("Przy {} komponentach składowych pozostało {:.2f}% początkowej wariancji zbioru".format(len(transformed_crypto_pca.columns), pca2.explained_variance_ratio_.sum()*100))
```

    Przy 6 komponentach składowych pozostało 93.57% początkowej wariancji zbioru
    


```python
features = range(pca2.n_components_)
plt.bar(features, pca2.explained_variance_ratio_, color='green')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()
```


    
![png](Cryptocurrencies_data_mining_files/Cryptocurrencies_data_mining_137_0.png)
    


### 9.3. Klastrowanie metodą k-średnich

Algorytm centroidów polega na znalezieniu liczby k-punktów będących "środkami ciężkości" centroidów w zbiorze danych. Środek ciężkości to lokalizacja reprezentująca środek pomiędzy wszystkimi pobliskimi punktami. Pierwsze środki ciężkości generujemy losowo i musi być ich tyle, na ile klastrów chcemy podzielić nasz zbiór danych, a w następnych krokach dokonujemy iteracyjne obliczenia w celu optymalizacji pozycji centroidów. Algorytm zatrzymuje swoje działanie, gdy centroidy (czyli *i* nie zmieniają się już, czyli ustabilizowały się, a także wówczas, gdy zdefiniowana liczba iteracji została osiągnięta. Warto zaznaczyć, że tego typu podejście nie wykrywa klastrów wklęsłych.

#### 9.3.1. Klastrowanie całego zbioru danych


```python
inertia = []
k = list(range(1, 16))
for i in k:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(crypto_scaled)
    inertia.append(km.inertia_)
```

Przyjęto liczbę klastrów (*n_clusters*) równą zmiennej *i*. W przykładzie powyższym wykorzystano inercję, czyli bezwładność, która zakłada, że powstałe klastry składają sie z grup wypukłych punktów. Inercja jest obliczana, jako odległość pomiędzy daną kolumną (wierszem, punktem), a średnią wartością dla kolumn (wierszy). Im większa inercja tym punkty są oddalone dalej od średniego profilu wiersza/kolumny. Słabo sprawdza się ona przy klastrach o bardziej wydłużonych i nieróżnorodnych kształtach.


```python
silhouttescore = []
ch_index = []
k = list(range(2, 16))
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42).fit(crypto_scaled)
    silhouttescore.append(metrics.silhouette_score(crypto_scaled, kmeans.labels_))
    ch_index.append(metrics.calinski_harabasz_score(crypto_scaled, kmeans.labels_))
```

Wykorzystano listę *silhouette_score* oraz *ch_index* do zapisania wyników współczynnika silhouette (średni współczynnik sylwetki) oraz indeksu CH wszystkich próbek. W obu przypadkach największa wartość definiuje najbardziej korzystną liczbę klastrów. Indeks CH został wyliczony na podstawie wzoru:
$$
CH = \frac{b}{a},
$$
 gdzie:  
 a - średnia odległość między punktami w klastrze do środka ciężkości danego klastra (spójność),  
 b - średnia ogległość między środkami ciężkości klastra od globalnego środka ciężkości (separacja).  

Indeks Calińskiego-Harabasza informuje o podobieństwie między obiektu do własnego skupienia względem innych skupień. Im większa wartość indeksu tym klastry są gęstrze i dobrze rozdzielone. Wyliczona wartość indeksu pozwala na określenie najbardziej korzystnej liczby klastrów oraz pozwala na porównanie algorytmów klastrowania między sobą.  
Wartość silhouette została wylicozna na podstawie wzoru:
$$
silhouette = \frac{(b-a)}{max(a,b)},
$$
 gdzie:  
 a - średnia odległość między punktami klastra,  
 b - średnia ogległość od najbliższego klastra dla każdego punktu.  

Z powyższego wzoru można zauważyć, że jeśli *a* będzie większe od *b*, to konkretna obserwacja ma dalej do obserwacji w swojej grupie (średnia odległość), a bliżej do obserwacji w sąsiedniej. Wówczas nasze równanie przyjmuje formę wzoru:
$$
\frac{(b-a)}{a} = \frac{b}{a} - 1,
$$
Skoro *a* było większe, to *b/a* będzie mniejsze niż 1, czyli całe równanie będzie mniejsze od 0. Analogicznie, jeśli *a* będzie mniejsze od *b* uzyska się wartość większą od 0.  
Z jednej strony, wartość *a* informuje o bliskości innych elementów wewnątrz grupy. Nie jest to zasadniczo gęstość, ale dostarcza informacji o cisno upakowanej grupie. Z kolei wartość *b* pokazuje, jak daleko znajduje się punkt od najbliższej innej grupy. Czyli mówi o separacji. Im większa, tym lepiej punkty są od siebie odeseparowane. Najlepiej więc, aby *a* było małe (blisko do „sojuszników”), i *b* duże (daleko do „wrogów”). Wyznaczenie maksymalnej wartości silhouette może pozwolić na poprawne określenie rozwiązania problemu separacja-koncentracja i w ten sposób umożliwia wyznaczenie najlepszego *k*. Można zaznaczyć, że algorytm informuje również o tym, czy dana funkcja grupująca poprawnie pogrupowała wartości. Jeśli współczynnik sylwetki wyjdzie ujemny, oznacza to, że uzyskano błędnie pogrupowane wartości.
 


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
    


#### 9.3.2. Klastrowanie zbioru zredukowanego przez PCA

W tym podejściu wykorzystujemy metodę redukcji wymiarowości PCA po której stosujemy algorytm k-średnich. Występuje zatem powtórzenie etapów poprzedniego klastrowania:


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
    


#### 9.3.3. Wybranie najlepszego modelu


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
    

Powyższe wartości miar pokazują, że model uzyskał lepszą skuteczność dla danych wcześniej zredukowanych poprzez PCA. W obu przypadkach wartości są większe niż model bez zredukowanych wymiarów. Ważną obserwacją jest brak spójności wyników modelu bez PCA (wykres z podpunktu 9.3.1). Wartość współczynnika Silhoutte wskazuje na optymalne rozwiązanie dla 14 klastrów natomiast wartość indeksu CH dla 6 klastrów. W przypadku modelu z redukcją wymiarów za pomocą metody PCA (wykres z podpunktu 9.3.2) największe wartości miar dobroci modelu zostały wyliczone w obu przypadkach dla 6 klastrów, co oznacza spójność wyników. Informacja ta jest kolejnym argumentem potwierdzającym wyższość modelu z redukcją wymiarowości niż bez. Dodatkowo na wykresie bezwładności (podpunkt 9.3.2) widoczny jest tzw. "łokieć", czyli miejsce, gdy wartość inercji spada wyraźnie wolniej. Dalszej analizie poddano jedynie model z 6 klastrami zaimplementowany na danych ze zmiejszoną wymiarowością. 

### 9.4. Przedstawienie wyników klasteryzacji

#### 9.4.1. Wyniki modelu końcowego

Dokonana zostanie klasteryzacja danych po przejściu przez metodę PCA, a następnie stworzonych zostanie 6 klastrów, które zostaną zaprezntowane na histogramie oraz zostanie przedstawiona tabela przypisująca daną kryptowlautę do odpowiedniego klastra.


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
    


Można zauważyć, że na powyższym histogramie do dwóch klastrów zostało przydzielone tylko po jednej walucie. Jest to kryptowaluta upadła oraz kryptowaluta niedawno powstała. Oby dwa tokeny w dużym stopniu odstają od pozostałych.


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



Kryptowaluty o największym kursie, wielkości rynku czy stopie zwrotu, jak Bitcoin oraz Ethereum, zostały przydzielone do klastra 0. Oznacza to, że model skutecznie wybrał liderów rynku do jeden grupy, gdyż znalazły się w niej dobrze prosperujące kryptowaluty, takie jak: LiteCoin, Cardano, Polkadot, Solana, BinanceCoin. Dużą zaletą modelu jest oddzielenie walut o bardzo niskim kursie, które zaliczyły upadek, tak jak terraclassic. Daje to jasną informację, że należy ich unikać.

#### 9.4.2. Wizualizacja wyników w 2D


```python
tsne = TSNE()
tsne_features = tsne.fit_transform(transformed_crypto_pca)
```

Dane zostają zredukowane do dwóch wymiarów za pomocą metody t-SNE. Jest to technika nieliniowej redukcji wymiarowości bazującej na podobieństwie między punktami. Metoda ta skupia się na zachowaniu lokalnej struktury danych i działa poprzez minimalizowanie odległości między punktami w gausie.


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



Dwuwymiarowy zbiór kryptowalut został podzielony na 6 klas.


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
    


Klastrowanie dwuwymiarowych danych jest zabiegiem czysto wizualnym i nie należy kierować się jego wynikami przy dokonywaniu decyzji inwestycyjnych. Pewna część informacji została jednak zachowana, dzięki czemu możemy zaobserwować, że największe kryptowaluty znajdują się w dwóch klastrach (na wykresie żółty i zielony). Tym razem model nie był juz w stanie znaleźć kryptowalut wyraźnie odstających od reszty, co podważa jego wiarygodność.
