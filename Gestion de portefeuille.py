#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install https://github.com/matplotlib/mpl_finance/archive/master.zip')
get_ipython().system('pip install pandas_datareader')


# In[4]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr   
import datetime

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from mpl_finance import candlestick_ohlc
import pylab


# In[5]:


start = datetime.datetime(2013,1,1)
end = datetime.date.today()

pylab.rcParams['figure.figsize'] = (15, 9) #taille des graphiques


# In[6]:


symbol_cac40={"lvmh":'MC.PA', "danone": 'BN.PA',"thales": 'HO.PA',"airbus": 'AIR.PA',"total": 'FP.PA',"veolia": 'VIE.PA',              "societegenerale": 'GLE.PA',"vinci": 'DG.PA',"peugeot": 'UG.PA',"capgemini": 'CAP.PA',"axa": 'CS.PA',"safran": 'SAF.PA',              "Airliquide": 'AI.PA',"carrefour": 'CA.PA',"orange": 'ORA.PA',"accor": 'AC.PA',"bouygues": 'EN.PA',"worldline": 'WLN.PA',              "kering": 'KER.PA',"engie": 'ENGI.PA',"BNP": 'BNP.PA',"creditagricole": 'ACA.PA',"sanofi": 'SAN.PA',"pernodricard": "RI.PA",              "schneiderelectric": 'SU.PA',"l_oreal": 'OR.PA',"michelin": 'ML.PA',"vivendi": 'VIV.PA',"atos": 'ATO.PA',"sodexo": 'SW.PA',              "legrand": 'LR.PA',"saintgobain": 'SGO.PA',"arcelormittal": 'MT.AS',"dassault": 'DSY.PA',"essilorluxottica": 'EL.PA',              "hermes": 'RMS.PA',"publicis": 'PUB.PA',"technipfmc": 'FTI.PA',"unibail": 'URW.AS',"renault": 'RNO.PA',"stmicroelectronics": 'STM.PA'}

name_cac40 = ["lvmh", "danone","thales","airbus","total","veolia","societegenerale","vinci","peugeot","capgemini","axa","safran","Airliquide",              "carrefour","orange","accor","bouygues","worldline","kering","engie","BNP","creditagricole","sanofi","pernodricard",              "schneiderelectric","l_oreal","michelin","vivendi","atos","sodexo","legrand","saintgobain","arcelormittal","dassault",              "essilorluxottica","hermes","publicis","technipfmc","unibail","renault","stmicroelectronics"]

cac40 = pdr.get_data_yahoo("^FCHI", start, end)
data_cac40 = {}

for x in name_cac40 : 
    data_cac40[x] = pdr.get_data_yahoo(symbol_cac40[x],start=start, end=end)


# In[7]:


data_cac40.keys()


# # Diagramme à bougies

# In[8]

def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader \
    from "yahoo"
    :param dat: un DataFrame de pandas avec une date type datetime64 et les colonnes de floats suivantes : "Open", "High", "Low", and "Close", \
    le plus souvent créées à partir du datareader de "yahoo"

    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", \
    "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param stick: une chaine de caractères ou un nombre pour la période de temps couverte par une seule bougie. Les seules chaines de caractères \
    possibles sont  "day", "week", "month", et "year", ("day" par défault), et tous les autres floats qui indiquent le nombre de trades dans la \
    période.

    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
    :param otherseries: un itérable de type liste qui contient les colonnes de dat pour les autres séries à plotter en lignes

    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    
    Sort un graphique en bougie pour les dates stockées dans dat. 
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12

    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365

    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))

    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')


    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)

    ax.grid(True)

    # Création du graphique en chandelier
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)

    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()
    
# In[9]

def candlestick(ticker, start, end):
    
    df_ohlc = pdr.get_data_yahoo(ticker, start, end)[['Open','High','Low','Close']]
    df_ohlc.reset_index(inplace=True)

    fig = go.Figure(data=[go.Candlestick(x=df_ohlc['Date'],
                    open=df_ohlc['Open'],
                    high=df_ohlc['High'],
                    low=df_ohlc['Low'],
                    close=df_ohlc['Close'])])
    fig.show()
    
# In[10]

candlestick('BNP.PA', start, end)

# # Représentation des données brutes

# In[11]

stocks = pd.DataFrame({"MC.PA": data_cac40["lvmh"]["Adj Close"],
                       "GLE.PA": data_cac40["societegenerale"]["Adj Close"],
                       })
stocks.head()

# In[12]

stocks.plot(grid = True)

# In[13]

stocks.plot(secondary_y = ["GLE.PA"], grid = True)

# In[14]

stock_return = stocks.apply(lambda x: np.log(x/x.shift(1)))
stock_return.loc['2020-03-01':].plot(grid = True).axhline(y = 0, color = "black", lw = 2)

# In[15]

def bollinger_bands(name, start, end):
    df = pdr.DataReader(name, 'yahoo', start, end)
   
    # On calcule la moyenne mobile 20, l'écart-type pour en déduire les bandes supérieures et inférieures
    df['20 Day MA'] = df['Adj Close'].rolling(window=20).mean()
    
    df['20 Day STD'] = df['Adj Close'].rolling(window=20).std() 
    df['Upper Band'] = df['20 Day MA'] + (df['20 Day STD'] * 2)
    df['Lower Band'] = df['20 Day MA'] - (df['20 Day STD'] * 2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_axis = df.index.get_level_values(0)
    
    # On remplit l'espace séparant les bandes supérieures et inférieures des bandes de Bollinger
    ax.fill_between(x_axis, df['Upper Band'], df['Lower Band'], color='grey')
    
    # On trace les prix ajustés de fermture et la moyenne mobile 20
    ax.plot(x_axis, df['Adj Close'], color='blue', lw=2)
    ax.plot(x_axis, df['20 Day MA'], color='r', lw=2)
    
    # Titre et présentation du graphe
    ax.set_title('20 Day Bollinger Band For {}'.format(name))
    ax.set_xlabel('Date (Year/Month)')
    ax.set_ylabel('Price(USD)')
    plt.grid(True)
    ax.legend() 
    
# In[16]

bollinger_bands('AAPL',start, end)
    
# # Problem du Stock Split et Dividendes

# In[17]:


def ohlc_adj(dat):
    """
    :param dat: pandas DataFrame de données brutes issues de Yahoo ("Open", "High", "Low", "Close", and "Adj Close")  "Adj Close" les prix de fermeture ajustés

    :return: pandas DataFrame avec les données ajustées

    Cette fonction règle le problème des splits, des dividendes etc. 
    Retourne un tableau semblable aux données de Yahoo, mais avec les données réelles. 
    """
    return pd.DataFrame({"Open": dat["Open"] * dat["Adj Close"] / dat["Close"],
                       "High": dat["High"] * dat["Adj Close"] / dat["Close"],
                       "Low": dat["Low"] * dat["Adj Close"] / dat["Close"],
                       "Close": dat["Adj Close"]})


# # Moyenne mobile

# ## Calcul des moyennes mobiles

# In[18]:


lvmh_adj = ohlc_adj(data_cac40["lvmh"])
lvmh_adj.head()


# In[19]:


lvmh_adj["20d"] = lvmh_adj["Close"].rolling(window = 20, center = False).mean()
lvmh_adj["50d"] = lvmh_adj["Close"].rolling(window = 50, center = False).mean()
lvmh_adj["200d"] = lvmh_adj["Close"].rolling(window = 200, center = False).mean()
lvmh_adj.dropna(inplace=True)


# # Trend Following Strategy

# In[20]:


lvmh_adj['20d-50d'] = lvmh_adj['20d'] - lvmh_adj['50d']

lvmh_adj["Regime"] = np.where(lvmh_adj['20d-50d'] > 0, 1, 0)
lvmh_adj["Regime"] = np.where(lvmh_adj['20d-50d'] < 0, -1, lvmh_adj["Regime"])

lvmh_adj["Regime"].value_counts()


# In[21]:


lvmh_adj["Regime"][-1] = 0
lvmh_adj["Signal"] = np.sign(lvmh_adj["Regime"] - lvmh_adj["Regime"].shift(1))
lvmh_adj.dropna(inplace=True)
lvmh_adj['Signal'].plot(figsize=(9,4))


# In[22]:


# on créé un tableau avec les opérations qui sont faites

lvmh_adj_signals = pd.concat([pd.DataFrame({"Price": lvmh_adj.loc[lvmh_adj["Signal"] == 1, "Close"],
                                            "Regime":lvmh_adj.loc[lvmh_adj["Signal"] == 1, "Regime"],
                                            "Signal": "Buy"}),
                              pd.DataFrame({"Price": lvmh_adj.loc[lvmh_adj["Signal"] == -1, "Close"],
                                            "Regime": lvmh_adj.loc[lvmh_adj["Signal"] == -1, "Regime"],
                                            "Signal": "Sell"}),])
lvmh_adj_signals.sort_index(inplace = True)

# J'enlève le premier ordre s'il s'agit d'une vente
if lvmh_adj_signals["Signal"][0] == "Sell" :
    lvmh_adj_signals.drop(lvmh_adj_signals.index[0], inplace = True)

lvmh_adj_signals.head()


# In[23]:


lvmh_adj_long_profits = pd.DataFrame({
    "Price": lvmh_adj_signals.loc[(lvmh_adj_signals["Signal"] == "Buy") & lvmh_adj_signals["Regime"] == 1, "Price"],
    
    "Profit": pd.Series(lvmh_adj_signals["Price"] - lvmh_adj_signals["Price"].shift(1)).loc[
        lvmh_adj_signals.loc[(lvmh_adj_signals["Signal"].shift(1) == "Buy") & (lvmh_adj_signals["Regime"].shift(1) == 1)].index
        ].tolist(),
    
    "End Date": lvmh_adj_signals["Price"].loc[
        lvmh_adj_signals.loc[(lvmh_adj_signals["Signal"].shift(1) == "Buy") & (lvmh_adj_signals["Regime"].shift(1) == 1)].index
        ].index
    })

lvmh_adj_long_profits.head()


# In[24]:


lvmh_adj_long_profits['Profit'].sum()

# In[25]:


tradeperiods = pd.DataFrame({"Start": lvmh_adj_long_profits.index,
                             "End": lvmh_adj_long_profits["End Date"]})

lvmh_adj_long_profits["Low"] = tradeperiods.apply(lambda x: np.min(lvmh_adj.loc[x["Start"]:x["End"], "Low"]), axis = 1)

lvmh_adj_long_profits.head()


# In[26]:




