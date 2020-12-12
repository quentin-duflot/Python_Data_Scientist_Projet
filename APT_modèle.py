# télécharger les données sur ce site : http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#International
# Données journalières qui renseigne de l'actualité macroéconomique dans plusieurs région du monde

"On importe les modules nécessaires au web scraping" 
import requests
import urllib.request
from bs4 import BeautifulSoup
"On récupère le lien du site que l'on souhaite scraper"
url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#International'
response = requests.get(url)
print (response) #Si 200 s'affiche cela signifie que l'accès au site a fonctionné

"On extrait le code source au format BeautifulSoup"
soup = BeautifulSoup(response.text, "html.parser")

"print(soup.findAll('a')) #La ligne qui nous intéresse est la 512"
target = soup.findAll('a')[512]
link = target['href']

"On extrait le tableau de données au format CSV"
download_url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/' + link
urllib.request.urlretrieve(download_url,'Europe_5_Factors_Daily_CSV.zip' )

#Créer une fonction qui retourne les coefficients de la régression linéaire simple et multiples de chaque facteur sur l'action étudié 
"On importe les modules nécessaires aux régressions linéaires"
from sklearn.linear_model import LinearRegression

def CoeffReg(X,Y) :
    beta = []
    modeleReg=LinearRegression()
    for i in range(len(Y)) :
        modeleReg.fit(X,Y[i])
        beta += modeleReg.coef
    modeleReg.fit(X,Y)
    BETA = modeleReg.coef
    return beta, BETA

#Créer une fonction qui calcule la valeur attendue de l'action avec la formule de l'APT.

def APT(d_start,m_start,y_start,d_end,m_end,y_end,action,PARAM):
    
    Rf = 0 #Peut-être modifié, correspond au taux du livret A
    DateBA = str(y_start+"-"+m_start+"-"+d_start)
    DateBP = str(y_start+m_start+d_start)
    DateEA = str(y_end+"-"+m_end+"-"+d_end)
    DateEP = str(y_end+m_end+d_end)
    X = df[df["Date"] in [DateBA:DateEA]] #df est à remplacer par le nom que l'on a donné au dataframe
    Y = df2[df2["Date"] in [DateBP:DateEP]] #df2 est à remplacer et Date est le nom à donner à la colonne du tableur CSV
    
    beta,BETA = CoeffReg(X, Y)
    ER = Rf + beta*BETA
    
    return ER
    