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

data_cac40 = {}

for x in name_cac40 : 
    data_cac40[x] = pdr.get_data_yahoo(symbol_cac40[x],start=start, end=end)


# In[7]:


data_cac40.keys()


# # Problem du Stock Split et Dividendes

# In[8]:


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

# In[9]:


lvmh_adj = ohlc_adj(data_cac40["lvmh"])
lvmh_adj.head()


# In[10]:


lvmh_adj["20d"] = lvmh_adj["Close"].rolling(window = 20, center = False).mean()
lvmh_adj["50d"] = lvmh_adj["Close"].rolling(window = 50, center = False).mean()
lvmh_adj["200d"] = lvmh_adj["Close"].rolling(window = 200, center = False).mean()
lvmh_adj.dropna(inplace=True)


# # Trend Following Strategy

# In[11]:


lvmh_adj['20d-50d'] = lvmh_adj['20d'] - lvmh_adj['50d']

lvmh_adj["Regime"] = np.where(lvmh_adj['20d-50d'] > 0, 1, 0)
lvmh_adj["Regime"] = np.where(lvmh_adj['20d-50d'] < 0, -1, lvmh_adj["Regime"])

lvmh_adj["Regime"].value_counts()


# In[12]:


lvmh_adj["Regime"][-1] = 0
lvmh_adj["Signal"] = np.sign(lvmh_adj["Regime"] - lvmh_adj["Regime"].shift(1))
lvmh_adj.dropna(inplace=True)
lvmh_adj['Signal'].plot(figsize=(9,4))


# In[13]:


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


# In[14]:


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


# In[15]:


lvmh_adj_long_profits['Profit'].sum()


# In[16]:


tradeperiods = pd.DataFrame({"Start": lvmh_adj_long_profits.index,
                             "End": lvmh_adj_long_profits["End Date"]})

lvmh_adj_long_profits["Low"] = tradeperiods.apply(lambda x: np.min(lvmh_adj.loc[x["Start"]:x["End"], "Low"]), axis = 1)

lvmh_adj_long_profits.head()


# In[ ]:




