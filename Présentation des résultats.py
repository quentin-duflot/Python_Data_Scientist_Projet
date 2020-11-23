def ma_crossover_orders(stocks, famille_fonction, start=end2,end=end):
    """
    :param stocks: liste de tuples (symbol, data brute téléchargée de Yahoo)
    :param famille_fonction : Dictionnaire contenant le Classifier de la famille à laquelle appartient l'action en index

    :return: pandas DataFrame contenant les ordres à passer
    """
    
    trades = pd.DataFrame({"Price": [], "Regime": [], "Signal": []})
    for s in stocks:
        s_redim = s[1][start:end].copy()
        clf = famille_fonction[s[0]]
        x = prepare(s_redim,start,end) #50 lignes de moins que s[1][start:end]
        x = x[['MACD', 'RSI', 'STO_K', "D", '20d-50d']]
        s_redim["Prédiction"] = np.concatenate((50*[-1], clf.predict(x))) #les 50premières dates sont vides à cause de la fct prepare

        signals = pd.concat([
            pd.DataFrame({"Price": s_redim.loc[s_redim["Prédiction"] == 2, "Adj Close"],
                         "Regime": s_redim.loc[s_redim["Prédiction"] == 2, "Prédiction"],
                         "Signal": "Acheter"}),
            pd.DataFrame({"Price": s_redim.loc[s_redim["Prédiction"] == 0, "Adj Close"],
                         "Regime": s_redim.loc[s_redim["Prédiction"] == 0, "Prédiction"],
                         "Signal": "Vendre"}),
        ])

        signals.index = pd.MultiIndex.from_product([signals.index, [s[0]]], names = ["Date", "Symbol"])
        trades = trades.append(signals)

    trades.sort_index(inplace = True)
    trades.index = pd.MultiIndex.from_tuples(trades.index, names = ["Date", "Symbol"])

    return trades
    
    
    

def backtest(stocks, cash,famille_fonction, portfolio = dict(),port_prices = dict()):
    """
    :param stocks: Une liste de tuples (Symbole de l'action, données brutes de Yahoo)
    :param cash: integer for starting cash value
    :param portfolio: Dictionnaire  nb d'actions détenues à t
    :param port_prices: Dictionnaire prix de l'action à t
    
    :return: pandas DataFrame contenant les ordres à passer

    On se servira dans cette fonction de l'historique de la valeur du portefeuille afin de voir comment il évolue dans le temps. 
    """
    signals = ma_crossover_orders(stocks,famille_fonction)

    # Dataframe qui contiendra les données historiques
    results = pd.DataFrame({"Cash Dispo": [],
                            "Portfolio Value": [],
                            "Type": [],
                            "Nb Action": [],
                            "Trade Value": [],
                            "Total Cash": []})

    for index, row in signals.iterrows():

        portfolio_val = 0 
        for key, val in portfolio.items():
            for i,dat in stocks :
               if key ==i:
                  port_prices[key] = dat.loc[str(index[0].date()),"Adj Close"]
            portfolio_val += val * port_prices[key]

        shares = portfolio[index[1]]
        old_price = port_prices[index[1]]
        
        n_max = np.floor((portfolio_val + cash)/(row["Price"]*len(portfolio)))

        if row["Signal"]=="Acheter" and shares<n_max : #la deuxième condition évite d'acheter deux fois consécutives
            trade_val = (n_max-shares)*row["Price"]
            portfolio[index[1]]= n_max
            port_prices[index[1]] = (shares*port_prices[index[1]]+(n_max-shares)*row["Price"])/(n_max)
 
            portfolio_val += trade_val
            cash -= trade_val

            # Update report
            results = results.append(pd.DataFrame({
                    "Cash Dispo":cash,
                    "Portfolio Value": portfolio_val,
                    "Type": row["Signal"],
                    "Nb Action": n_max-shares,
                    "Trade Value": trade_val,
                    "Total Cash": cash +portfolio_val
                }, index = [index]))

        elif row["Signal"]=="Vendre" and shares > 0:
            trade_val = shares * row["Price"]
            portfolio[index[1]]= 0
            port_prices[index[1]] = row["Price"]

            portfolio_val -= trade_val
            cash += trade_val

            # Update report
            results = results.append(pd.DataFrame({
                    "Cash Dispo":cash,
                    "Portfolio Value": portfolio_val,
                    "Type": row["Signal"],
                    "Nb Action": shares,
                    "Trade Value": trade_val,
                    "Total Cash": cash +portfolio_val
                }, index = [index]))

        else : pass 

    results.sort_index(inplace = True)
    results.index = pd.MultiIndex.from_tuples(results.index, names = ["Date", "Symbol"])
    return results