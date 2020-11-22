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

    return risk_free_return + beta*(data["m_returns"].mean()*12-risk_free_return)

capm(lvmh, cac40, "01 01 2016", "14 10 2019")