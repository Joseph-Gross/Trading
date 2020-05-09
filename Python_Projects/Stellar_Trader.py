#!/usr/bin/env python
# coding: utf-8




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
crypto_list = ['BTC','XTZ','XLM','ETH','XRP','EOS','LTC','BCH','OXT','ATOM','ETC','LINK','REP','ZRX','ALGO','KNC']
not_listed_cryptos = ['XTZ','ALGO','OXT','ATOM']





def ljust(sarr):
    tempsarr = []
    for i in sarr:
        tempsarr.append(i.ljust(9))
    
    return tempsarr

def makeBuySellIndex(df):
    
    sell_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    buy_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]

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




currency = 'XLM'
while True:
    while True: 
        try:
            starttime_data = time.time()
            starttime = time.time()
            data = pd.DataFrame(client.get_product_historic_rates(currency + '-USD',granularity=3600)) 
            data.columns = ['Time','Low','High','Open','Close','Volume']
            data.sort_values(by = 'Time',ascending = True, inplace = True)
            data = data.iloc[-70:,:]
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
        




            print()

            if (list(data['MACD Line'])[-1] < list(data['Signal Line'])[-1]) & (data['Slow Stoch'].tolist()[-1]<20):

                if data['RSI']<30:
                    trade_size_buy = str(round(float(client.get_accounts()[20]['balance'])))
                else:
                    trade_size_buy = 10
                try:    
                    client.place_market_order(product_id='XLM-USD',side='buy', funds=trade_size_buy)  
                    print('$' + str(trade_size_buy) + ' Bought')
                except:
                    print('Buy Error')
            else:
                print('No Buy Opportunity')


            if (data['RSI Dipped'].tolist()[-1] & (data['MACD Line'].tolist()[-1]>data['Signal Line'].tolist()[-1])&(data['RSI'].tolist()[-1]>40)):
                trade_size_sell = str(round(float(client.get_accounts()[22]['balance'])))

                try:
                    client.place_market_order(product_id='XLM-USD',side='sell', funds=trade_size)  
                    print(str(trade_size_sell) + ' XLM Sold')

                except:
                    print('Sell Error')
            else:
                print('No Sell Opportunity')

            print()
            print('Calc Time'.ljust(12)  + str(round(time.time()-starttime_data,3)).ljust(6,'0'))
            print('MACD'.ljust(12)       + str(round(data['MACD Line'].tolist()[-1]*10000, 3)).ljust(6,'0'))
            print('Signal'.ljust(12)     + str(round(data['Signal Line'].tolist()[-1]*10000, 3)).ljust(6,'0'))
            print('Slow Stoch'.ljust(12) + str(round(data['Slow Stoch'].tolist()[-1], 3)).ljust(6,'0'))
            print('RSI'.ljust(12)        + str(round(data['RSI'].tolist()[-1], 3)).ljust(6,'0'))

            print()

            while (time.time()-starttime) < 3600:
                print(str(datetime.datetime.now().hour).rjust(2,'0') + ':' + str(datetime.datetime.now().minute).rjust(2,'0') + ' -- ' +str(round(float(client.get_product_ticker(product_id='XLM-USD')['price']),4)))
                pag.move(10,10)
                time.sleep(60)
                pag.move(-10,-10)
        except:
            print(' _____________________ ')
            print('|                     |')
            print('|*********************|')
            print('|******* ERROR *******|')
            print('|*********************|')
            print('|_____________________|')
        











