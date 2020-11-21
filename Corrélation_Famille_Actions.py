
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


family = [[] for k in range(41)]


liste_actions = []
for corr in mat_corr:
  liste_actions.append(corr)
liste_actions


S = 41

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
  X[i] = data[i][['MACD', 'RSI', 'STO_K', 'D', '20d-50d']]
  Y[i] = data[i]["Signal"]
