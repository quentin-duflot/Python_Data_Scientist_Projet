
# On récupère les cours ajusté des actions du cac40 et on regarde la matrice de corrélation

stocks_cac40 = pd.DataFrame({x : data_cac40[x]["Adj Close"] for x in name_cac40})

f = plt.figure(figsize=(15, 12))
plt.matshow(stocks_cac40.corr(), fignum=f.number)
plt.xticks(range(stocks_cac40.shape[1]), stocks_cac40.columns, fontsize=9.5, rotation=45)
plt.yticks(range(stocks_cac40.shape[1]), stocks_cac40.columns, fontsize=9.5)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title('Matrice de corrélation', fontsize=30)