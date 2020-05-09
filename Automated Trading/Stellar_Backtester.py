#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
client = cbpro.AuthenticatedClient('f50f329e7b723a33d290aef45c5bf87d', 
                                   'w4kvUc5/+k3njZZlld8N2Z1f1+fnLg029bXwmx8cYFA2dD0hUNDiuWjbAKHm+PdFQZ9+aSJbRi17vsCK4QPN3Q==', '1bs1qjxc3f9')


# In[ ]:


crypto_list = ['BTC','XTZ','XLM','ETH','XRP','EOS','LTC','BCH','OXT','ATOM','ETC','LINK','REP','ZRX','ALGO','KNC']
not_listed_cryptos = ['XTZ','ALGO','OXT','ATOM']
data_dic = {}


# In[2]:


def ljust(sarr):
    tempsarr = []
    for i in sarr:
        tempsarr.append(i.ljust(9))
    
    return tempsarr

def makeBuySellIndex(df):
    
    sell_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    buy_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]

    return {'buy':buy_ind,'sell':sell_ind}



def getdata(symbol,interval = '1h',daysback = 5,unix = True):
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


# In[69]:


for curr in ['XTZ','ALGO','LINK','OXT', 'LTC']:

    if curr not in not_listed_cryptos:
        tempdf = getdata(interval = '1h',symbol = curr,daysback = 730,unix = True)
    else:
        tempdf = pd.DataFrame(client.get_product_historic_rates(curr + '-USD',granularity=3600)) 
        tempdf.columns = ['Time','Low','High','Open','Close','Volume']
    tempdf = tempdf.sort_values(by = 'Time',ascending = True).reset_index()
    tempdf['Gain'] = tempdf['Close']>tempdf['Open']
    tempdf['Change'] = tempdf['Close']-tempdf['Open']
    tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
    tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
    tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
    tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
    tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
    tempdf['RSI'] = makeRSI(tempdf)
    tempdf['Fast Stoch'] = makeStoch(tempdf,3)
    tempdf['Slow Stoch'] = makeStoch(tempdf,14)
    tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
    tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
    tempdf['Stoch Dipped'] = RSIDipped(tempdf['Fast Stoch'])
    tempdf['Stoch Peaked'] = RSIPeaked(tempdf['Fast Stoch'])
    
    data_dic[curr] = tempdf


# In[35]:




for curr in ['XLM','BTC','ETH','XRP','EOS']:

    if curr not in not_listed_cryptos:
        tempdf = getdata(interval = '1h',symbol = curr,daysback = 730,unix = True)
    else:
        tempdf = pd.DataFrame(client.get_product_historic_rates(curr + '-USD',granularity=3600)) 
        tempdf.columns = ['Time','Low','High','Open','Close','Volume']
    tempdf = tempdf.sort_values(by = 'Time',ascending = True).reset_index()
    tempdf['Gain'] = tempdf['Close']>tempdf['Open']
    tempdf['Change'] = tempdf['Close']-tempdf['Open']
    tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
    tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
    tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
    tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
    tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
    tempdf['RSI'] = makeRSI(tempdf)
    tempdf['Fast Stoch'] = makeStoch(tempdf,3)
    tempdf['Slow Stoch'] = makeStoch(tempdf,14)
    tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
    tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
    tempdf['Stoch Dipped'] = RSIDipped(tempdf['Fast Stoch'])
    tempdf['Stoch Peaked'] = RSIPeaked(tempdf['Fast Stoch'])
    
    data_dic[curr] = tempdf


# In[91]:


currency = 'XLM'
position_size = 1000
fee_size = .007


data = data_dic[currency]
bs_index = makeBuySellIndex(data)
buy_index   = bs_index['buy']
sell_index = bs_index['sell']


sell_index = sell_index[sell_index < len(data)-2]
buy_index = buy_index[buy_index < sell_index[-1]]

metric = 'Close'



profit = []
real_profit = []
place = -1
real_buy_ind = []
real_sell_ind = []
time_held = []
# print('--- Trades History ---')
for i in buy_index:
    if sell_index[sell_index>i][0] > place:

        buy_price = data['Close'][i]
        sell_price = data['Close'][sell_index[sell_index>i][0]]
        
        real_buy_ind.append(i)
        real_sell_ind.append(sell_index[sell_index>i][0])
        time_held.append(sell_index[sell_index>i][0] - i)
        place = sell_index[sell_index>i][0]
        profit.append(sell_price-buy_price)
#         if sell_price<buy_price:
#             print('Buy: ' + str(i).ljust(8) + 'Sell: ' + str(sell_index[sell_index>i][0]).ljust(8) + 'Profit: ' + str(((sell_price-buy_price)/buy_price)*position_size))
#         else:
#             print('Buy: ' + str(i).ljust(8) + 'Sell: ' + str(sell_index[sell_index>i][0]).ljust(8) + 'Profit:  ' + str(((sell_price-buy_price)/buy_price)*position_size))

        real_profit.append(((sell_price-buy_price)/buy_price )* position_size)
profit = np.array(profit)
real_profit = np.array(real_profit)

print()
col1 = []
col2 = []
col1.append('Profit with $'+str(position_size)+' Positions:')
col2.append(str(round(pd.Series(real_profit).sum(),4)))
col1.append('Transaction Fees: ')
col2.append(str(round(position_size*fee_size*len(profit),4)))
col1.append('Trades Made:')
col2.append(str(len(profit)))
col1.append('Success Rate w/o Fees: ')
col2.append(str(round(100*(len(profit[profit>0])/len(profit)),4)) + '%')

col1.append('Success Rate w/ Fees: ')

col2.append(str(round(100*(len(real_profit[(pd.Series(real_profit)-(position_size*fee_size))>0])/len(real_profit)),4)) + '%')

col1.append('Average Gain: ')
col2.append(str(round(stat.mean(pd.Series(real_profit[real_profit>0])),4)))
col1.append('Average Loss: ')
col2.append(str(round(stat.mean(pd.Series(real_profit[real_profit<0])),4)))
col1.append('Net Profit: ' )
col2.append(str(round(pd.Series(real_profit).sum() - (position_size*fee_size*len(profit)),4)))
col1.append('Average Time Held: ' )
col2.append(str(stat.mean(time_held)))






clrs = []
for i in real_profit:
    if i >0:
        clrs.append('green')
    else:
        clrs.append('red')

plt.figure(figsize = (20,7))
plt.style.use('dark_background')
plt.plot(data.index,data[metric],color = 'b',label = metric,linewidth = 0)
plt.scatter(data.index[real_buy_ind],data[metric][real_buy_ind],label = 'Buy',color = 'Green',s = .5)
plt.scatter(data.index[real_sell_ind],data[metric][real_sell_ind],label = 'Sell',color = 'Red', s = .5)
plt.title(currency + ': Buy Sell')
plt.legend()
plt.grid(True,alpha = .3)
plt.show()
        

        
plt.figure(figsize = (20,7))
plt.style.use('dark_background')
plt.bar(list(range(0,len(real_profit))),real_profit, color = clrs)
plt.grid(True,alpha = .3)
plt.title(currency + ': Historical Gains w/o Fees')
plt.show()
plt.close()

plt.figure(figsize = (20,7))
plt.style.use('dark_background')
plt.bar(list(range(0,len(real_profit))),pd.Series(real_profit) - (position_size*fee_size), color = clrs)
plt.grid(True,alpha = .3)
plt.title(currency + ': Historical Gains w/ Fees')
plt.show()
plt.close()


print()
account_values = []
for i in range(len(real_profit)):
    account_values.append(sum(real_profit[0:i])-(position_size*fee_size))


plt.figure(figsize = (20,7))
plt.style.use('dark_background')
plt.plot(list(range(len(profit))),(pd.Series(account_values)+1000)-(position_size*fee_size), color = 'w')
plt.grid(True,alpha = .3)
plt.title(currency + ': Account Values Over Trade History')
plt.show()
plt.close()


show_recent_trades = 20
plt.figure(figsize = (20,7))
plt.plot(list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):],list(data['Close'])[(data.index[real_buy_ind][-show_recent_trades]):],color = 'w',linewidth = .6,alpha = .6)
plt.scatter(list(data.index[real_buy_ind])[-show_recent_trades:],list(data['Close'][real_buy_ind])[-show_recent_trades:],label = 'Buy',color = 'Green',s = 25)
plt.scatter(list(data.index[real_sell_ind])[-show_recent_trades:],list(data['Close'][real_sell_ind])[-show_recent_trades:],label = 'Sell',color = 'Red', s = 25)
plt.title(currency + ': Last ' + str(show_recent_trades) + ' Trades')
plt.xlim(list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):][0]-1,list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):][-1]+5)
plt.grid(True,alpha = .3)
plt.show()
plt.close()

print()
stats = pd.DataFrame(ljust(col2),col1)
stats.columns = ['']
stats




