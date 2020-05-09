#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import pandas as pd
import yfinance as yf
import pyautogui as pag
import time 
import alpaca_trade_api as tradeapi
import statistics as stat
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from yahoo_fin import stock_info as si
import datetime
import scipy.interpolate as interpolate
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.signal import find_peaks
import cbpro
import requests
import numpy as np
register_matplotlib_converters()


# In[2]:


client = cbpro.AuthenticatedClient('f50f329e7b723a33d290aef45c5bf87d', 
                                   'w4kvUc5/+k3njZZlld8N2Z1f1+fnLg029bXwmx8cYFA2dD0hUNDiuWjbAKHm+PdFQZ9+aSJbRi17vsCK4QPN3Q==', '1bs1qjxc3f9')


# In[3]:


crypto_list = ['BTC','XTZ','XLM','ETH','XRP','EOS','LTC','BCH','OXT','ATOM','ETC','LINK','REP','ZRX','ALGO','KNC']
not_listed_cryptos = ['XTZ','ALGO','OXT','ATOM']


# In[107]:


def makeBuySellIndex(df):
    buy_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    sell_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]


    return {'buy':buy_ind,'sell':sell_ind}



def getdata(symbol,interval = '1d',daysback = 5,unix = True):
    symbol = symbol.upper()
    dta = yf.download(tickers = symbol + '-USD', period = str(daysback)+'d' , interval = interval).reset_index()
    if unix:
        dta['Time'] = pd.DatetimeIndex (dta['Date']).astype ( np.int64 )/1000000
    else:
        dta['Time'] = pd.to_datetime(dta['Date'])
    dta.drop(['Date'],axis = 1,inplace = True)
    return dta    
def makeTimes(unix):
    date = datetime.datetime.utcfromtimestamp(unix)
    return date
def makeRSI(df):
    period = 7
    rsi_list = []
    for i in range(period,len(df)):
        tempdata = df[i-period:i]  
        if len(tempdata[tempdata['Gain']]['Change'])==0:
            gains = 0
        else:          
            gains = stat.mean(tempdata[tempdata['Gain']]['Change'])    
        if len(tempdata[tempdata['Gain']==False]['Change'])==0:
            loses = 0
        else:    
            loses = stat.mean(tempdata[tempdata['Gain']==False]['Change'].apply(abs))
        if loses == 0:
            rsi_list.append(100)
        else:    
            rsi_list.append(100 - 100/(1+(gains/loses)))
            
    return [0]*period + rsi_list
def makeOBV(df):
    period = 10
    obv_list = []
    for i in range(period,len(df)):
        tempdata = df[i-period:i]
        gains = sum(tempdata[tempdata['Gain']]['Change'])
        loses = sum(tempdata[tempdata['Gain']==False]['Change'])
        obv_list.append(gains+loses)
    return [0]*period + obv_list
def makeSMA(Series,period):
    sma_list = []
    closes = list(Series)
    for i in range(period,len(Series)): 
        sma_list.append(stat.mean(closes[i-period:i]))
    return [0]*period + sma_list
def makeStDevLst(df):
    period = 10
    closes = df['Close'].apply(float).tolist()
    stdev_list = []
    for i in range(period,len(df)):
        stdev_list.append(stat.pstdev(closes[i-period:i]*2)*2)
    return [0]*period + stdev_list
def laterPrice(period,df):
    temp_list = df['Close'].shift(-period).tolist()[0:-period] 
    return temp_list + [-1]*period
def makeStoch(df,period):
    stoch_list_raw = []
    stoch_list = []
    closes = df['Close'].apply(float).tolist()
    highs = df['High'].apply(float).tolist()
    lows = df['Low'].apply(float).tolist()
    for i in range(period,len(df)):
        high = max(highs[i-period:i+1])
        low = min(lows[i-period:i+1])
        stoch_list_raw.append(((closes[i]-low)/(high-low) )*100)
    for i in range(3,len(stoch_list_raw)):
        stoch_list.append(stat.mean(stoch_list_raw[i-3:i]))
    stoch = makeSMA(stoch_list,3)
    return [0]*(3+period) + stoch_list   
def makeEMA(s,n):
    s = s.tolist()
    ema = []
    j = 1
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)
    ema.append(( (s[n] - sma) * multiplier) + sma)
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return [0]*(n-1) + ema
def RSIDipped(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-7:i]<30]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp
def RSIPeaked(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-7:i]>70]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp
def StochDipped(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]<20]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp
def StochPeaked(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]>80]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp

def makeBuySellGraph(currency, plot_history = False):
    currency = currency.upper()
    price = client.get_product_ticker(currency + '-USD')
    
    if currency not in not_listed_cryptos:
        data = getdata(interval = '1d',symbol = currency,daysback = 1000,unix = True)
    else:
        data = pd.DataFrame(client.get_product_historic_rates(currency + '-USD',granularity=86400)) 
        data.columns = ['Time','Low','High','Open','Close','Volume']
    data = data.sort_values(by = 'Time',ascending = True).reset_index()
    data['Gain'] = data['Close']>data['Open']
    data['Change'] = data['Close']-data['Open']
    data['Percent Change'] = ((data['Close']-data['Open'])/data['Open'])*100
    data['12 EMA'] = pd.DataFrame(data['Close']).ewm(span=12, min_periods=12).mean().values
    data['26 EMA'] = pd.DataFrame(data['Close']).ewm(span=26, min_periods=26).mean().values
    data['MACD Line'] = [0]*25+((data['12 EMA']-data['26 EMA']).tolist()[25:])
    data['Signal Line'] = pd.DataFrame(data['MACD Line']).ewm(span=15, min_periods=15).mean().values
    data['RSI'] = makeRSI(data)
    data['Fast Stoch'] = makeStoch(data,3)
    data['Slow Stoch'] = makeStoch(data,14)
    data['RSI Dipped'] = RSIDipped(data['RSI'])
    data['RSI Peaked'] = RSIPeaked(data['RSI'])
    data['Stoch Dipped'] = RSIDipped(data['Fast Stoch'])
    data['Stoch Peaked'] = RSIPeaked(data['Fast Stoch'])
    
    
    bs_index = makeBuySellIndex(data)
    buy_index   = bs_index['buy']
    sell_index = bs_index['sell']


    sell_index = sell_index[sell_index < len(data)-2]
    buy_index = buy_index[buy_index < sell_index[-1]]

    metric = 'Close'

    plt.figure(figsize = (20,7))
    plt.style.use('dark_background')
    plt.plot(data.index,data[metric],color = 'b',label = metric,linewidth = 1)
    plt.scatter(data.index[buy_index],data[metric][buy_index],label = 'Buy',color = 'Green')
    plt.scatter(data.index[sell_index],data[metric][sell_index],label = 'Sell',color = 'Red')
    plt.title(currency + ': Buy Sell')
    plt.legend()
    plt.grid(True,alpha = .3)
    plt.show()
    
    profit = []
    real_profit = []
    place = -1
    print('--- Trades History ---')
    for i in buy_index:
        if sell_index[sell_index>i][0] > place:

            buy_price = data['Close'][i]
            sell_price = data['Close'][sell_index[sell_index>i][0]]
            place = sell_index[sell_index>i][0]
            profit.append(sell_price-buy_price)
            if sell_price<buy_price:
                print('Buy: ' + str(i).ljust(8) + 'Sell: ' + str(sell_index[sell_index>i][0]).ljust(8) + 'Profit: ' + str(((sell_price-buy_price)/buy_price)*200))
            else:
                print('Buy: ' + str(i).ljust(8) + 'Sell: ' + str(sell_index[sell_index>i][0]).ljust(8) + 'Profit:  ' + str(((sell_price-buy_price)/buy_price)*200))

            real_profit.append(((sell_price-buy_price)/buy_price )* 200)
    profit = pd.Series(profit)
    print('--- Stats ---')
    print('Average Profit Per Coin:',stat.mean(profit))
    print('Trades Made:',len(profit))
    print('Total Profit with $200 Positions:', pd.Series(real_profit).sum())
    
    if plot_history:
        clrs = []
        for i in real_profit:
            if i >0:
                clrs.append('green')
            else:
                clrs.append('red')
        
        plt.figure(figsize = (20,7))
        plt.style.use('dark_background')
        plt.bar(list(range(0,len(real_profit))),real_profit, color = clrs)
        plt.grid(True,alpha = .3)
        plt.title(currency + ': Historical Gains')
        plt.show()
        print()
        print()
        print()
    
def makeMetricGraph(metric,currency,plot_metric = True,plot_orders = True):
    if currency not in not_listed_cryptos:
        data = getdata(interval = '1d',symbol = currency,daysback = 1000,unix = True)
    else:
        data = pd.DataFrame(client.get_product_historic_rates(currency + '-USD',granularity=86400)) 
        data.columns = ['Time','Low','High','Open','Close','Volume']
    data = data.sort_values(by = 'Time',ascending = True).reset_index()
    data.dropna(inplace = True)
    data['Gain'] = data['Close']>data['Open']
    data['Change'] = data['Close']-data['Open']
    data['Percent Change'] = ((data['Close']-data['Open'])/data['Open'])*100
    data['12 EMA'] = pd.DataFrame(data['Close']).ewm(span=12, min_periods=12).mean().values
    data['26 EMA'] = pd.DataFrame(data['Close']).ewm(span=26, min_periods=26).mean().values
    data['MACD Line'] = [0]*25+((data['12 EMA']-data['26 EMA']).tolist()[25:])
    data['Signal Line'] = pd.DataFrame(data['MACD Line']).ewm(span=15, min_periods=15).mean().values
    data['RSI'] = makeRSI(data)
    data['Fast Stoch'] = makeStoch(data,3)
    data['Slow Stoch'] = makeStoch(data,14)
    data['RSI Dipped'] = RSIDipped(data['RSI'])
    data['RSI Peaked'] = RSIPeaked(data['RSI'])
    data['Stoch Dipped'] = RSIDipped(data['Fast Stoch'])
    data['Stoch Peaked'] = RSIPeaked(data['Fast Stoch'])
    
    lookback = 20
    data['Previous Percent Change'] = [0]*lookback + (((data['Close'][lookback:] - data['Close'].shift(lookback)[lookback:])/data['Close'].shift(lookback)[lookback:])*100).tolist()
    
    bs_index = makeBuySellIndex(data)
    buy_index   = bs_index['buy']
    sell_index = bs_index['sell']
    sell_index = sell_index[sell_index < len(data)-2]
    buy_index = buy_index[buy_index < sell_index[-1]]
    
    if plot_metric|plot_orders:
        plt.figure(figsize = (20,7))
        plt.style.use('dark_background')
        if plot_metric:
            plt.plot(data.index,data[metric],color = 'b',alpha = .75)
        if plot_orders:
            plt.scatter(data.index[buy_index],data[metric][buy_index],label = 'Buy',color = 'Green')
            plt.scatter(data.index[sell_index],data[metric][sell_index],label = 'Sell',color = 'Red')
        plt.title(currency + ' : ' + metric)
        plt.legend()
        plt.grid(True,alpha = .3)
        plt.show()
    
    plt.figure(figsize = (20,3))
    sns.distplot(data[metric].values[50:-1])
    plt.title(currency)
    plt.yticks([0],[''])
    plt.xlim(-75,75)
    plt.grid(True)
    plt.show()


# In[108]:


c = []
ap = []
tp = []
app = []
t = []
tp200 = []

for currency in crypto_list:
    
    currency = currency.upper()
    price = client.get_product_ticker(currency + '-USD')
    
    if currency not in not_listed_cryptos:
        data = getdata(interval = '1d',symbol = currency,daysback = 1000,unix = True)
    
    else:
        print('Yahoo Data Not Available For ' + currency)
        data = pd.DataFrame(client.get_product_historic_rates(currency + '-USD',granularity=86400)) 
        data.columns = ['Time','Low','High','Open','Close','Volume']


    data = data.sort_values(by = 'Time',ascending = True).reset_index()
    data['Gain'] = data['Close']>data['Open']
    data['Change'] = data['Close']-data['Open']
    data['Percent Change'] = ((data['Close']-data['Open'])/data['Open'])*100
    data['12 EMA'] = pd.DataFrame(data['Close']).ewm(span=12, min_periods=12).mean().values
    data['26 EMA'] = pd.DataFrame(data['Close']).ewm(span=26, min_periods=26).mean().values
    data['MACD Line'] = [0]*25+((data['12 EMA']-data['26 EMA']).tolist()[25:])
    data['Signal Line'] = pd.DataFrame(data['MACD Line']).ewm(span=15, min_periods=15).mean().values
    data['RSI'] = makeRSI(data)
    data['Fast Stoch'] = makeStoch(data,3)
    data['Slow Stoch'] = makeStoch(data,14)
    data['RSI Dipped'] = RSIDipped(data['RSI'])
    data['RSI Peaked'] = RSIPeaked(data['RSI'])
    data['Stoch Dipped'] = RSIDipped(data['Fast Stoch'])
    data['Stoch Peaked'] = RSIPeaked(data['Fast Stoch'])
    
    bs_index = makeBuySellIndex(data)
    buy_index   = bs_index['buy']
    sell_index = bs_index['sell']
    sell_index = sell_index[sell_index < len(data)-2]
    buy_index = buy_index[buy_index < sell_index[-1]]

    profit = []
    percent_profit = []
    p200 = []
    place = -1

    for i in buy_index:
        if sell_index[sell_index>i][0] > place:

            buy_price = data['Close'][i]
            sell_price = data['Close'][sell_index[sell_index>i][0]]
            place = sell_index[sell_index>i][0]
            profit.append(sell_price-buy_price)
            percent_profit.append(((sell_price-buy_price)/buy_price)*100)
            p200.append(((sell_price-buy_price)/buy_price)*200)


    profit = pd.Series(profit).to_numpy()
    c.append(currency)
    t.append(len(profit))
    if len(profit)>0:
        ap.append(stat.mean(profit))
    else:
        ap.append(0)
    tp.append(profit.sum())
    if len(percent_profit)>0:
        app.append(stat.mean(percent_profit))
    else:
        app.append(0)
    tp200.append(sum(p200))

performance  = pd.DataFrame({'Currency':c,'Average Percent Profit':app,'Average Profit':ap,'Total Profit':tp,
                             '$200 Positions Profit':tp200,'Trades':t}).sort_values(by = '$200 Positions Profit',
                                                      ascending = False).reset_index().drop(['index'],axis = 1)


# In[109]:


print(performance.head(len(performance)))
print()
print()


# In[110]:


print('This strategy started at $' + str(len(crypto_list)*200) + ' and ended at $' + str((len(crypto_list)*200) + performance['$200 Positions Profit'].sum()))
print('Total Profit:' + str(performance['$200 Positions Profit'].sum()))
print()
print()


# In[111]:


# plt.figure(figsize = (10,5))
# plt.bar(list(range(len(performance.groupby('Trades').mean()))), performance.groupby('Trades').mean()['$200 Positions Profit'].values,color = 'b')
# plt.plot([0,17],[0]*2, color = 'w')
# plt.xticks(list(range(len(performance.groupby('Trades').mean()))), performance.groupby('Trades').mean().index)
# plt.show()


# In[113]:


# for i in performance['Currency'].tolist():
#     makeBuySellGraph(i, plot_history = False)


# In[ ]:


# for i in crypto_list: 
#     makeMetricGraph('Previous Percent Change',i,plot_metric = False,plot_orders = True)
    


# In[ ]:





