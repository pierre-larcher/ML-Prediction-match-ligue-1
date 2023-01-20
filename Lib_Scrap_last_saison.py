"""" Description.

Cette librairie permet de récupérer les données de ligue 1 de la dernière saison.
Après changement du code source html par le site effectiué début janvier 2023.
"""

from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import pandas as pd
from pandas import NA
import csv


url = "https://www.oddsportal.com/soccer/france/ligue-1/results/"


def setup(url: str) -> webdriver:
    """Mise en place des éléments pour le scraping, setup du webdriver et lancement de la page."""
    option = webdriver.ChromeOptions()
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_argument("--profile-directory=Default")
    # REMPLACER LE USERNAME AVANT UTILISATION
    option.add_argument(
        "--user-data-dir=C:/Users/pierr/AppData/Local/Google/Chrome/User Data"
    )
    # Re-size la taille de la fenêtre
    option.add_argument("window-size=1920,1000")
    # Ajoute un useragent différent
    driver = webdriver.Chrome(options=option)
    driver.get(url)
    driver.maximize_window()
    return driver


def _recup_DF(driver: webdriver) -> pd.DataFrame:

    Cotes = driver.find_elements(
        By.CLASS_NAME, "flex-center.flex-col.gap-1.pt-1.pb-1.border-black-borders"
    )
    Score = driver.find_elements(By.CLASS_NAME, "hidden.flex-col.items-center")
    Equipe = driver.find_elements(By.CLASS_NAME, "relative.block.truncate")
    Nb_BM = driver.find_elements(
        By.CLASS_NAME, "flex-col.items-center.justify-center.gap-1.border-l.pt-1.pb-1"
    )
    Année = driver.find_elements(By.CLASS_NAME, "flex.flex-wrap.gap-2.py-3.text-xs")

    Annee = []
    for i in range(0, len(Année)):
        try:
            Annee.append(Année[i].text)
        except:
            Annee.append(NA)

    Nb = []
    for i in range(0, len(Nb_BM)):
        Nb.append(Nb_BM[i].text)

    Cote = []
    for i in range(0, len(Cotes)):
        Cote.append(Cotes[i].text)

    score = []
    for i in range(0, len(Score)):
        score.append(Score[i].text)

    Score_D = []
    Score_E = []

    for i in range(0, len(score) - 2):
        Score_D.append(score[i][0])
        Score_E.append(score[i][2])

    D = []
    N = []
    E = []
    for i in range(0, len(Cote), 3):
        D.append(Cote[i])
    for i in range(1, len(Cote), 3):
        N.append(Cote[i])
    for i in range(2, len(Cote), 3):
        E.append(Cote[i])

    equipe = []
    for i in range(0, len(Equipe)):
        equipe.append(Equipe[i].text)

    Equipe_D = []
    Equipe_E = []

    for i in range(0, len(equipe), 2):
        Equipe_D.append(equipe[i])
    for i in range(1, len(equipe), 2):
        Equipe_E.append(equipe[i])

    Saison = []
    for i in range(0, len(Equipe_D)):
        Saison.append(Annee[0].split("\n")[0].replace("/", "-"))

    data = {
        "Année": Saison,
        "Cote_A": D,
        "Cote_null": N,
        "Cote_B": E,
        "Nb_BM": Nb,
        "Equipe_A": Equipe_D,
        "Equipe_B": Equipe_E,
        "Score_A": Score_D,
        "Score_B": Score_E,
    }
    return pd.DataFrame(data)


def _final(url: str) -> pd.DataFrame:
    """Execute toutes les fonctions"""
    driver = setup(url)
    _banner(driver)
    time.sleep(3)
    df = _recup_DF(driver)
    driver.close()
    return df


def _banner(driver: webdriver):
    """Accepte les cookies"""
    try:
        driver.find_elements(By.ID, "onetrust-accept-btn-handler").click()
    except:
        pass


def _derniere_page(driver: webdriver) -> int:
    """Permet de récupérer le nombre de page de la saison en cours"""
    caracteres = "\n"
    nb_page = driver.find_elements(By.XPATH, '//*[@id="pagination"]')
    page = []
    for i in nb_page[0].text:
        page.append(i)
    for caractere in caracteres:
        for i in page:
            if i == caractere:
                page.remove(i)
    for i in page:
        list(map(int, page))

    page_finale = int(max(page))
    return page_finale


def _page(url: str) -> int:
    """permet de set up le nombre de page de la saison"""
    option = webdriver.ChromeOptions()
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_argument("--profile-directory=Default")
    driver = webdriver.Chrome(options=option)
    driver.get(url)
    _banner(driver)
    page_finale = _derniere_page(driver)
    return page_finale


def crea_DF_derniere_saison() -> csv:
    """Crée le data frame final en ittérant sur l'url (dernière saison url change)"""
    df_final = pd.DataFrame(
        columns=[
            "Année",
            "Cote_A",
            "Cote_null",
            "Cote_B",
            "Nb_BM",
            "Equipe_A",
            "Equipe_B",
            "Score_A",
            "Score_B",
        ]
    )
    for i in range(2022, 2023):
        page_finale = _page(
            "https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/1/"
        )
        for j in range(1, page_finale + 1):
            df_temporaire = _final(
                "https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/{}/".format(
                    j
                )
            )
            df_final = pd.concat([df_final, df_temporaire])
    df_final.reset_index(drop=True, inplace=True)
    df_final["Score_A"] = df_final["Score_A"].astype(str)
    df_final["Score_B"] = df_final["Score_B"].astype(str)
    df_final["Score_A"] = df_final.Score_A.apply(lambda x: x[0])
    df_final["Score_B"] = df_final.Score_B.apply(lambda x: x[0])
    return df_final.to_csv(
        r"Extraction2bis.csv", index=False, header=True, sep=";", encoding="utf-8-sig"
    )
