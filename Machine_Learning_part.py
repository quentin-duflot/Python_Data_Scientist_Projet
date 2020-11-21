#1) Calculer des signaux d'achat que l'on juge correct pour servir de modèle pour un futur algorithme de trading : Préparation Feature Y



#2) On prépare les features X, en séparant les données. Une partie pour l'entrainement (70%) et l'autre pour le test (30%)

#3) On applique un Random Forest Classifier sur nos données, et on visualise les résultats : Accuracy, Matrice de Confusion, et répartition de l'importance de chaque feature.


### importation des modules nécessaires

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import sys
from random import uniform


### 1) Préparation Feature Y 

#Calculer des signaux d'achat entre start et end que l'on juge corrects pour servir de modèle pour un futur algorithme de trading : Préparation Feature Y


def training_set(data, results, start, end, alpha=0.0):
    """
    Acheter : 2 ; Conserver : 1 ; Vendre : 0
    """
    d = pd.Timedelta('1 day')
    if len(results[start:end]) < 4:
        try:
            if data.loc[end, "Close"] > data.loc[start, "Close"]:
                results.loc[start, "Signal"]= 2
            else :
                results.loc[end, "Signal"]= 2
        except KeyError: #erreur si start n'est pas un jour ouvré
            pass
        return results
t = []
    for index, row in data[end-7*d:end].iterrows():
        t.append(index)
    end_0 = t[-1]   # correspond à la dernière date du tableau
    end_1 = t[-2]   # avant dernière date 
    end_2 = t[-3]   # avant avant dernière date 
    
    i_m = data.loc[end_2:end_0,"Close"].idxmin()
    i_M = data.loc[end_2:end_0,"Close"].idxmax()
    m = data.loc[i_m, "Close"]
    M = data.loc[i_M, "Close"]
    if abs(m-M)>alpha*m : #condition de profit
        if (i_m== end_2 and i_M == end_0) or (i_m== end_0 and i_M == end_2) :
            results.loc[end_2:end_0,"Signal"] = 1  #on conserve car monotone sur l'intervalle de taille 3
            return  training_set(data, results, start, end_1)

        elif i_m == end_1: #le minimum est le point du milieu , inversion de tendance
            results.loc[i_m,"Signal"] = 2 
            return  training_set(data, results, start, end_1)
        elif i_M == end_1: 
            results.loc[i_M,"Signal"] = 0 
            return  training_set(data, results, start, end_1)
  
    else :  # pas de profit ici
        results.loc[end_2:end_0,"Signal"] = 1 #on conserve car pas de profit réalisable
        return  training_set(data, results, start, end_1)
        


### Préparation Feature X

#On télécharge et prépare les features X, en séparant les données. Une partie pour l'entrainement (70%) et l'autre pour le test (30%)


def prepare(df,start=start, end = end,N=14):
    
    df = ohlc_adj(df[start:end]) #Cours ajusté

    df["20d"] = np.round(df["Close"].rolling(window = 20, center = False).mean(), 2)
    df["50d"] = np.round(df["Close"].rolling(window = 50, center = False).mean(), 2) 
    df["20d-50d"] = df["20d"] - df["50d"]                                             
    df = calcul_MACD(df)
    df["e9"] = pd.Series.ewm(df['MACD'], span=12).mean()

    df["K"] = 100 *(df["Close"]-df["Low"].rolling(window = N).min())/(df["High"].rolling(window = N).max()- df["Low"].rolling(window = N).min())
    df["D"] = df["K"].rolling(window=3).mean()
    df["momentum"] = df["Close"] - df["Close"].shift(12)

    results = pd.DataFrame({"Price" : df["Close"].copy(),
                          "Signal" : 1})
    results = training_set(df,results,start,end,0) 

    data = pd.DataFrame({"MACD" : df["MACD"]-df["e9"],
                    "RSI" : RSI_mobile(df, N),
                    "STO_K" : df["K"],
                    "D" : df["D"],
                    "20d-50d" : df["20d-50d"],
                    "momentum" : df["momentum"],
                    "Signal" : results["Signal"]
                    })[50:] 
    #on ne prend pas les 50 premières lignes car la colonne "20d-50d" contient NaN : pas applicable avec Random Forest
    
    
    
    return (data)

