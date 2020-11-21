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
