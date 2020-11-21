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