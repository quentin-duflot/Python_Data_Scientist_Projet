import pandas_datareader as pdr
from datetime import date
import numpy as np
import pandas as pd


risk_free_return = 0.05 #taux du livret A par exemple

start= "01 01 2015"
end = date.today()

lvmh = pdr.get_data_yahoo("MC.PA", start, end)
cac40 = pdr.get_data_yahoo("^FCHI", start, end) #marché sur lequel se trouve lvmh : cac40

def capm(df1, df2, start, end):
    """
    :param df1: pandas DataFrame de données brutes issues de Yahoo ("Open", "High", "Low", "Close", and "Adj Close")
    :param df2: pandas DataFrame de données brutes issues de Yahoo ("Open", "High", "Low", "Close", and "Adj Close")
    :param start: str de la première date de la période
    :param end: str de la dernière date de la période
    
    :return: beta, CAPM
    
    Calcule le beta et le CAPM sur une période, à la date end. 
    CAPM = r_f + beta * (market_return - r_f)
    
    r_f : taux d'un placement sans risque : livret A en France
    market_retun : rentabilité esperée du marché : rentabilité historique 
    beta : beta de l'actif financier df1

    """

    stock1 = df1[start:end]
    stock2 = df2[start:end]

    #On garde les lignes à une période d'un mois
    return_s1 = stock1.resample('M').last()
    return_s2 = stock2.resample('M').last()
    
    data = pd.DataFrame({'s_adjclose' : return_s1['Adj Close'], 'm_adjclose': return_s2['Adj Close']}, index=return_s1.index)
    
    data[['s_returns','m_returns']] = np.log(data[['s_adjclose', 'm_adjclose']]/data[['s_adjclose', 'm_adjclose']].shift(1))
    
    data = data.dropna() #on enlève les valeurs nulles

    covmat = np.cov(data["s_returns"], data["m_returns"])
    beta = covmat[0,1]/covmat[1,1]

    return beta,risk_free_return + beta*(data["m_returns"].mean()*12-risk_free_return)

capm(lvmh, cac40, "01 01 2016", "14 10 2019")


def tab_capm(df, start,end,N):
  d = pd.Timedelta('1 day')
  res = []

  while start + 70*d != end:
      try :
          s = df.loc[str(end)]
          res.append(capm(df, cac40, max(end - N*d, start) ,end)[1])
      except KeyError :
          pass
      end = end -d 
  
  for i in range(71):
    try :
        s = df.loc[str(start+i*d)]
        res.append(0)
    except KeyError :
        pass
  
  return np.array(res)[::-1]
   

  


    