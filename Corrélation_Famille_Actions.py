
# On récupère les cours ajusté des actions du cac40 et on regarde la matrice de corrélation

stocks_cac40 = pd.DataFrame({x : data_cac40[x]["Adj Close"] for x in name_cac40})

f = plt.figure(figsize=(15, 12))
plt.matshow(stocks_cac40.corr(), fignum=f.number)
plt.xticks(range(stocks_cac40.shape[1]), stocks_cac40.columns, fontsize=9.5, rotation=45)
plt.yticks(range(stocks_cac40.shape[1]), stocks_cac40.columns, fontsize=9.5)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title('Matrice de corrélation', fontsize=30)

cac ={x : clf for x in name_cac40}


# On crée deux matrices de corrélation 
# L'une nous sert à sélectionner les différentes actions selon leur corrélation
mat_corr = stocks_cac40.corr().copy()

# L'autre est immuable, et permet de ne sélectionner qu'une seule et unique fois chaque action
mat_corr_immu = stocks_cac40.corr().copy()


# liste qui contiendra les différentes familles
family = [[] for k in range(41)]


# On crée une liste avec le nom des actiosn pour relier un indice à un nom d'action
liste_actions = []
for corr in mat_corr:
  liste_actions.append(corr)
liste_actions

# Compteur pour n'avoir qu'une famille par action
S = 41

# Dictionnaire des indices de la matrice de corrélation immuable qu'on videra au fur et à mesure qu'une action sera choisie  (une liste verrait ses indices modifiés)

liste_index = {k:k for k in range(0,S)}

for num_ligne in range(0,S):
  if not num_ligne in liste_index:
    continue 
  for num_colonne in range(0,S):
    if not num_colonne in liste_index:
      continue
    if abs(mat_corr_immu.iloc[num_ligne, num_colonne]) > 0.8 :
      family[num_ligne].append(liste_actions[num_colonne])
      mat_corr.drop([liste_actions[num_colonne]], axis = 'columns', inplace = True)
      del liste_index[num_colonne]


# Extraction des familles sous forme de DataFrame

family2 = []
for k in range(41):
  if family[k] != []:
    family2.append(family[k])
    

end2 = datetime.datetime(2018,10,31)
data = {k:[] for k in range(len(family2))}
X = {k:k for k in range(len(family2))}
Y = {k:k for k in range(len(family2))}

for i in range(len(family2)):
  data[i] = pd.concat([prepare(data_cac40[family2[i][j]] , end=end2) for j in range(len(family2[i]))])
  data[i] = data_augmentation(data[i])
  X[i] = data[i][['MACD', 'RSI', 'STO_K', 'D', '20d-50d', 'momentum']]
  Y[i] = data[i]["Signal"]

### Construction d'un clf par famille 

#Séparation des données en données d'entrainement(70%) et de données de test(30%) 
donnee_test = []
cac2 = cac.copy()
for i in range(len(family2)):
  donnee_test.append(train_test_split(X[i], Y[i], test_size=0.3))
  
#donnee_test : 1er indice i : famille i
# 2ème indice j : 0 : X_train ; 1 : X_test ; 2 : Y_train ; 3 : Y_test

#Création de la fonction avec Random Forest, une par famille
foret_alea = [] #fonction de la famille i à l'indice i
y_pred = []
for i in range(len(family2)):
  clf=RandomForestClassifier(min_samples_leaf = 1, min_samples_split = 2, n_estimators=1500) # n : nombre d'arbres
  clf.fit(donnee_test[i][0],donnee_test[i][2])
  y_pred.append(clf.predict(donnee_test[i][1]))
  # on met à jour le classifier pour les actions correspondantes
  for act in family2[i]:
    cac2[act] = clf
    



# Exemple pour la famille 1 (indice 0)

print("Accuracy:",metrics.accuracy_score(donnee_test[0][3], y_pred[0]))

feature_imp = pd.Series(cac['lvmh'].feature_importances_, index = ["MACD",'RSI', 'STO_K', "D", '20d-50d', 'momentum']).sort_values(ascending=False)

#Matrice de confusion 

metrics.plot_confusion_matrix(foret_alea[0],donnee_test[0][1], donnee_test[0][3])
#Ligne : y_test   ; Colonne : y_pred

# Mesure l'importance des features avec un histogramme
sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel("Scores de l'importance des features dans dans notre modèle")
plt.ylabel('Features')
plt.title("Visualisation de l'importance des features pour la famille 1", fontsize=30)
plt.legend()
plt.show()
