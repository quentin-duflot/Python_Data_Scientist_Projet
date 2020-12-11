# télécharger les données sur ce site : http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#International
# Données journalières qui renseigne de l'actualité macroéconomique dans plusieurs région du monde

"On importe les modules nécessaires au web scraping 
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

