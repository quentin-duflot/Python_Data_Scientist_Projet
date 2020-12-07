# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:41:45 2020

@author: User
"""
N = 14

lvmh_adj["K"] = 100 *(lvmh_adj["Close"]-lvmh_adj["Low"].rolling(window = N).min())/(lvmh_adj["High"].rolling(window = N).max()- lvmh_adj["Low"].rolling(window = N).min())
lvmh_adj["D"] = lvmh_adj["K"].rolling(window=N).mean()

stochastique = pd.DataFrame({"K" : lvmh_adj["K"],
                             "D" : lvmh_adj["D"]})

stochastique.loc['2019-10-02':end,:].plot(grid=True).axhline(y = 20, color = "black", lw = 2)
plt.axhline(y=80, color = 'black', lw = 2)

lvmh_adj["K-80"] = lvmh_adj["K"] - 80
lvmh_adj["K-50"] = lvmh_adj["K"] - 50
lvmh_adj["K-20"] = lvmh_adj["K"] - 20
lvmh_adj["K-D"] = lvmh_adj["K"] - lvmh_adj["D"]

lvmh_adj.dropna(inplace=True)
lvmh_adj.head()

#Note 0 : acheter et 5 : vendre

def test_tab_bool(t1,t2):
  t= t1.copy()
  for i in range(len(t1)):
    t[i] = t1[i] and t2[i]
  return t

# on met 5 si K <20 et K traverse D à la hausse

t = np.where(test_tab_bool(test_tab_bool((lvmh_adj[N+1:]["K-D"][1:]>0),(lvmh_adj[15:]["K-D"].shift(1)[1:]<0)), (lvmh_adj[15:]["K-20"].shift(1)[1:]<0)) ,5,2.5)
lvmh_adj["Note"] = np.concatenate((np.array((N+2)*[-1.]),t))


# on met 0 si K>80 et K traverse D à la baisse
t_2 = np.where(test_tab_bool((lvmh_adj[N+1:]["K-D"][1:]<0),test_tab_bool((lvmh_adj[N+1:]["K-D"].shift(1)[1:]>0),(lvmh_adj[N+1:]["K-80"].shift(1)[1:]>0))),0,lvmh_adj[N+2:]["Note"])
lvmh_adj["Note"] = np.concatenate((np.array((N+2)*[-1.]) ,t_2))

lvmh_adj["Note"].value_counts()


def ma_crossover_orders(stocks, bas,haut, N):
    """
    :param stocks: Une liste de tuples (Symbole de l'action, données brutes de Yahoo) 
    :param bas: Int pour le seuil minimum que le stochastique peut franchir avant de considérer l'action en sur-vente
    :param haut: Int pour le seuil maximum que le stochastique peut franchir avant de considérer l'action en sur-achat
    :param N: Int pour la période du stochastique
    :return: pandas DataFrame contenant les ordres à passer

    Cette fonction détermine quand chaque action doit être vendue ou achetée suivant la méthode du stochastique.  
    Pour chaque opération elle renvoie d'autres informations comme le couurs de l'action à ce moment. 
    """
    trades = pd.DataFrame({"Price": [], "Regime": [], "Signal": []})
    for s in stocks:
        s[1]["K"] = 100 *(s[1]["Close"]-s[1]["Low"].rolling(window = N).min())/(s[1]["High"].rolling(window = N).max()- s[1]["Low"].rolling(window = N).min())
        s[1]["D"] = s[1]["K"].rolling(window=N).mean()

        s[1]["K-"+str(haut)] = s[1]["K"] - haut
        s[1]["K-50"] = s[1]["K"] -50
        s[1]["K-"+str(bas)] = s[1]["K"] - bas
        s[1]["K-D"] = s[1]["K"] - s[1]["D"]

        # on met 5 si K <20 et K traverse D à la hausse
        t = np.where(test_tab_bool(test_tab_bool((s[1][N+1:]["K-D"][1:]>0),(s[1][N+1:]["K-D"].shift(1)[1:]<0)), (s[1][N+1:]["K-20"].shift(1)[1:]<0)) ,5,2.5)
        s[1]["Note"] = np.concatenate((np.array((N+2)*[-1.]),t))

        # on met 0 si K>80 et K traverse D à la baisse
        t_2 = np.where(test_tab_bool((s[1][N+1:]["K-D"][1:]<0),test_tab_bool((s[1][N+1:]["K-D"].shift(1)[1:]>0),(s[1][N+1:]["K-80"].shift(1)[1:]>0))),0,s[1][N+2:]["Note"])
        s[1]["Note"] = np.concatenate((np.array((N+2)*[-1.]) ,t_2))

        # Get signals
        signals = pd.concat([
            pd.DataFrame({"Price": s[1].loc[s[1]["Note"] == 5, "Close"],
                         "Regime": s[1].loc[s[1]["Note"] == 5, "Note"],
                         "Signal": "Acheter"}),
            pd.DataFrame({"Price": s[1].loc[s[1]["Note"] == 0, "Close"],
                         "Regime": s[1].loc[s[1]["Note"] == 0, "Note"],
                         "Signal": "Vendre"}),
        ])

        signals.index = pd.MultiIndex.from_product([signals.index, [s[0]]], names = ["Date", "Symbol"])
        trades = trades.append(signals)

    trades.sort_index(inplace = True)
    trades.index = pd.MultiIndex.from_tuples(trades.index, names = ["Date", "Symbol"])

    return trades

def backtest(stocks, cash, portfolio = dict(),port_prices = dict()):
    """
    :param stocks: Une liste de tuples (Symbole de l'action, données brutes de Yahoo)
    :param cash: integer pour la valeur initiale investie
    :param portfolio: dictionnaire avec les actions détenues
    :param port_prices: dictionnaire avec le prix des actions détenues

    :return: pandas DataFrame avec des résultats d'un backtest de la stratégie

    On se servira dans cette fonction de l'historique de la valeur du portefeuille afin de voir comment il évolue dans le temps. 
    """

    signals = ma_crossover_orders(stocks, 20, 80,14)

    # Dataframe that will contain backtesting report
    results = pd.DataFrame({"Start Cash": [],
                            "End Cash": [],
                            "Portfolio Value": [],
                            "Type": [],
                            "Shares": [],
                            "Share Price": [],
                            "Trade Value": [],
                            "Profit per Share": [],
                            "Total Profit": []})

    for index, row in signals.iterrows():

        #valeur du portefeuille avant l'opération
        portfolio_val = 0 
        for key, val in portfolio.items():
            for i,dat in stocks :
                if key ==i:
                    port_prices[key] = dat.loc[str(index[0].date()),"Close"]
            portfolio_val += val * port_prices[key]

        shares = portfolio[index[1]]
        old_price = port_prices[index[1]]
        
        n_max = np.floor((portfolio_val + cash)/(row["Price"]*len(portfolio)))

        if row["Signal"]=="Acheter" and shares<n_max :
            trade_val = (n_max-shares)*row["Price"]
            portfolio[index[1]]=n_max
            port_prices[index[1]] = (shares*port_prices[index[1]]+(n_max-shares)*row["Price"])/(n_max)
            # Update report
            results = results.append(pd.DataFrame({
                    "Start Cash": cash,
                    "End Cash": cash - trade_val,
                    "Portfolio Value":cash + portfolio_val,
                    "Type": row["Signal"],
                    "Shares": n_max - shares,
                    "Share Price": row["Price"],
                    "Trade Value": trade_val,
                    "Profit per Share": old_price - port_prices[index[1]],
                    "Total Profit": (n_max - shares) * (old_price - port_prices[index[1]])
                }, index = [index]))
            cash -= trade_val

    
        elif row["Signal"]=="Vendre" and shares > 0:
            trade_val = shares * row["Price"]
            portfolio[index[1]]= 0
            port_prices[index[1]] = row["Price"]
            # Update report
            results = results.append(pd.DataFrame({
                    "Start Cash": cash,
                    "End Cash": cash + trade_val,
                    "Portfolio Value":cash +portfolio_val,
                    "Type": row["Signal"],
                    "Shares": shares ,
                    "Share Price": row["Price"],
                    "Trade Value": trade_val,
                    "Profit per Share": port_prices[index[1]],
                    "Total Profit": shares* (port_prices[index[1]] - old_price)
                }, index = [index]))
            
            cash += trade_val

        else : pass 

    results.sort_index(inplace = True)
    results.index = pd.MultiIndex.from_tuples(results.index, names = ["Date", "Symbol"])
    return results

end3 = datetime.datetime(2018,12,20)
stocks = [("MC.PA",ohlc_adj(data_cac40["lvmh"])[end3:end]),
                              ("BN.PA",ohlc_adj(data_cac40["danone"])[end3:end]),
                              ("HO.PA",ohlc_adj(data_cac40["thales"])[end3:end]),
                              ("AIR.PA",ohlc_adj(data_cac40["airbus"])[end3:end]),
                              ("VIE.PA",ohlc_adj(data_cac40["veolia"])[end3:end]),
                              ("DG.PA",ohlc_adj(data_cac40["vinci"])[end3:end]),
                              ("UG.PA",ohlc_adj(data_cac40["peugeot"])[end3:end]),
                              ("CAP.PA",ohlc_adj(data_cac40["capgemini"])[end3:end]),
                              ("CS.PA",ohlc_adj(data_cac40["axa"])[end3:end]),
                              ("SAF.PA",ohlc_adj(data_cac40["safran"])[end3:end])]

portfolio = {"MC.PA" : 0, "BN.PA" : 0,"HO.PA":0, "AIR.PA":0,"VIE.PA":0,"DG.PA":0,"UG.PA":0,"CAP.PA":0,"CS.PA":0,"SAF.PA":0}
port_prices = {"MC.PA" : 0, "BN.PA" : 0,"HO.PA":0, "AIR.PA":0,"VIE.PA":0,"DG.PA":0,"UG.PA":0,"CAP.PA":0,"CS.PA":0,"SAF.PA":0}

bk_sto = backtest(stocks, 100000, portfolio, port_prices) #commenté pour gagner du temps à compiler

bk_sto["Portfolio Value"].groupby(level = 0).apply(lambda x: x[-1]).plot(grid=True)
plt.title("Résultat du stochastique sur une période d'un an", fontsize=20)

bk["Start Cash"].groupby(level = 0).apply(lambda x: x[-1]).plot(grid=True) 
plt.title("Evolution du cash disponible : celui qui n'est pas investi", fontsize=20)