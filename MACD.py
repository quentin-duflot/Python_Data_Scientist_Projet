# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:23:08 2020

@author: User
"""
# On utilise la commande de pandas pandas.Series.ewm pour calculer des fonctions exponentielles pondérées
# Création de la fonction qui va calculer la MACD
def calcul_MACD(df):
    
    df['e26'] = pd.Series.ewm(df['Close'], span=26).mean() #Calcul de la moyenne mobile exponentielle 26
    df['e12'] = pd.Series.ewm(df['Close'], span=12).mean() #Calcul de la moyenne mobile exponentielle 12
    df['MACD'] = df['e12'] - df['e26'] # Calcul de la MACD
    df['Signal line'] = pd.Series.ewm(df['MACD'], span=9).mean() #Calcul de la ligne de signal
    return df

# On applique cette fonction à notre DataFrame avec les données ajustées
calcul_MACD(lvmh_adj).head()

#Calcul du croisement de la ligne de MACD et de la ligne de signal
lvmh_adj['MACD-Signal'] = lvmh_adj['MACD']-lvmh_adj['Signal line']
lvmh_adj.head()

# Visualisation des croisements de la ligne de MACD et de la ligne de signal :
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

# Visualisation de la ligne de MACD et du signal
ax1.plot(lvmh_adj.loc['2020-03-01':,'MACD'], label = 'MACD')
ax1.plot(lvmh_adj.loc['2020-03-01':,'Signal line'], label = 'Signal', c='red')
ax1.grid(True)
ax1.legend()
ax1.set_title("Ligne de MACD et de signal", fontsize=20)

ax2.plot(lvmh_adj.loc['2020-03-01':, 'MACD-Signal'], label = "MACD - MME9")
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("Différence entre MACD et Signal", fontsize=20)
ax2.axhline(y=0, c='black')

# On prend un nouveau dataframe pour garder lvmh_adj clean
very_new_lvmh_adj=lvmh_adj.copy()
very_new_lvmh_adj.head()

# Create a function to signal when to buy and sell an asset
def buy_sell(df):
  Buy = []
  Sell = []
  flag = -1

  for i in range(0, len(df)):
    if df['MACD'][i] > df['Signal line'][i] :
      Sell.append(np.nan)
      if flag != 1: #première fois qu'on passe par là
        Buy.append(df['Close'][i])
        flag = 1
      else :
        Buy.append(np.nan)
    elif df['MACD'][i] < df['Signal line'][i] :
      Buy.append(np.nan)
      if flag != 0: #première fois qu'on passe par là
        Sell.append(df['Close'][i])
        flag = 0
      else :
        Sell.append(np.nan)
    else :
      Buy.append(np.nan)
      Sell.append(np.nan)
  
  return Buy, Sell

very_new_lvmh_adj['Buy_Signal_Price'] = buy_sell(very_new_lvmh_adj)[0]
very_new_lvmh_adj['Sell_Signal_Price'] = buy_sell(very_new_lvmh_adj)[1]
very_new_lvmh_adj.head()

#Visually show the stock buy and sell signal
plt.figure()
plt.scatter(very_new_lvmh_adj.index, very_new_lvmh_adj['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
plt.scatter(very_new_lvmh_adj.index, very_new_lvmh_adj['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
plt.plot(very_new_lvmh_adj['Close'], label = 'Close Price', alpha = 0.35)

plt.title('Close Price Buy & Sells Signals')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend(loc = 'upper left')
plt.grid(True)