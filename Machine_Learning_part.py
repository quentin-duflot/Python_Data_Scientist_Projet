#1) Calculer des signaux d'achat que l'on juge correct pour servir de modèle pour un futur algorithme de trading : Préparation Feature Y

#2) Pas assez de données sur les valeurs Achat/Vente comparé au nombre de données sur la valeur Conserver. On fait de la Data Augmentation pour avoir les valeurs en nombre similaire. 


#3) On prépare les features X, en séparant les données. Une partie pour l'entrainement (70%) et l'autre pour le test (30%)

#4) On applique un Random Forest Classifier sur nos données, et on visualise les résultats : Accuracy, Matrice de Confusion, et répartition de l'importance de chaque feature.


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
        
### Data Augmentation

#Pas assez de données sur les valeurs Achat/Vente par rapport au nombre de données sur la valeur Conserver. On fait de la Data Augmentation pour avoir les valeurs en nombre similaire.

def data_augmentation(df,n=3,p=0.01):
  """
  La valeur de p est très importante : c'est le pourcentage de modification. (Accuracy = 70% avec  p=10%, =90% avec p=1%)
  Objectif de cette fonction est de multiplier par 4 environ le nombre de valeur achat et vente
  Donc à chaque fois qu'il y a une ligne dans df qui donne un signal vente ou achat, on en génère 3 autres. 
  Les 3 là doivent être des petites perturbations de la ligne initiale. 
  cette fonction sera appliquée à data à l'endroit indiqué ci dessous. à ce moment data a 0 : pour vendre, 1 pour conserver, 2 pour acheter

  :param df: DataFrame avec les colonnes des indicateurs et une colonne avec le signal calculé par training_set
  :param p: pourcentage de perturbation. 

  :return: DataFrame     comme df mais avec plus de lignes 
  """
  
  res = df.copy()
  d = pd.Timedelta('1 day')
  i=0
  for index, row in df.iterrows():
      if row["Signal"]==0 or row["Signal"]==2:
          for j in range(n):
              res = pd.concat([res, pd.DataFrame({"MACD" : row[0]*(1+uniform(-p,p)),
                                        "RSI" : row[1]*(1+uniform(-p,p)),
                                        "STO_K" : row[2]*(1+uniform(-p,p)),
                                        "D" : row[3]*(1+uniform(-p,p)),
                                        "20d-50d" :row[4]*(1+uniform(-p,p)),
                                        "momentum" : row[5]*(1+uniform(-p,p)),
                                        "capm" : row[6]*(1+uniform(-p,p)),
                                        "Signal" : row["Signal"]
                                        },index = [end + i*d])])
          i+=n
  return res

### Préparation Feature X

#On télécharge et prépare les features X, en séparant les données. Une partie pour l'entrainement (70%) et l'autre pour le test (30%)


def prepare(df,start=start, end = end,N=14):
    """
    Permet de généraliser la préparation des données à n'importe quelle action, entrée en paramètre. 

    :param df: data brute téléchargée depuis Yahoo 

    :return: DataFrame avec les features X et Y(colonne "Signal")
    """
    df1 = df.copy()
    df = ohlc_adj(df[start:end]) #Cours ajusté

    df["20d"] = np.round(df["Close"].rolling(window = 20, center = False).mean(), 2)
    df["50d"] = np.round(df["Close"].rolling(window = 50, center = False).mean(), 2) 
    df["20d-50d"] = df["20d"] - df["50d"]                                             
    df = calcul_MACD(df)
    df["e9"] = pd.Series.ewm(df['MACD'], span=12).mean()

    df["K"] = 100 *(df["Close"]-df["Low"].rolling(window = N).min())/(df["High"].rolling(window = N).max()- df["Low"].rolling(window = N).min())
    df["D"] = df["K"].rolling(window=3).mean()
    df["momentum"] = df["Close"] - df["Close"].shift(12)
    df["capm"] = tab_capm(df1, start, end, 400)

    results = pd.DataFrame({"Price" : df["Close"].copy(),
                          "Signal" : 1})
    results = training_set(df,results,start,end,0) 

    data = pd.DataFrame({"MACD" : df["MACD"]-df["e9"],
                    "RSI" : RSI_mobile(df, N),
                    "STO_K" : df["K"],
                    "D" : df["D"],
                    "20d-50d" : df["20d-50d"],
                    "momentum" : df["momentum"],
                    "capm" : df["capm"],
                    "Signal" : results["Signal"]
                    })[50:] 
    #on ne prend pas les 50 premières lignes car la colonne "20d-50d" contient NaN : pas applicable avec Random Forest
    
    return (data)

end2 = datetime.datetime(2018,10,31)
axa = pdr.get_data_yahoo('FP.PA',start=start, end=end2)
data = prepare(axa,end=end2)

# ici on applique la fonction data_augmentation(data)
data2 = data_augmentation(data)

X = data2[['MACD', 'RSI', 'STO_K', 'D', '20d-50d','momentum','capm']]
Y = data2["Signal"]

### Application du Random Forest


#Séparation des données en données d'entrainement(70%) et de données de test(30%) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#Création de la fonction avec Random Forest

clf=RandomForestClassifier(min_samples_leaf = 1, min_samples_split = 2, n_estimators=1500) # n : nombre d'arbres

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_, index = ["MACD",'RSI', 'STO_K', "D", '20d-50d','momentum','capm']).sort_values(ascending=False)

#Matrice de confusion 

metrics.plot_confusion_matrix(clf, X_test, y_test)

#Ligne : y_test   ; Colonne : y_pred

# Mesure l'importance des features avec un histogramme
sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel("Scores de l'importance des features dans dans notre modèle")
plt.ylabel('Features')
plt.title("Visualisation de l'importance des features", fontsize=30)
plt.legend()
plt.show()




