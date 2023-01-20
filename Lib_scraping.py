"""" Description.

Cette librairie permet de récupérer toutes les données de la ligue 1(côtes, équipes, buts.. )
puis les enregistre en csv. Nous avons utilisé selenium afin d'effectuer le scraping
"""


from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from pandas import NA
import csv


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


def recup_DF(driver: webdriver) -> pd.DataFrame:
    """Récupère les données et crée un data frame"""
    Rencontres = driver.find_elements(
        By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[2]'
    )
    Score = driver.find_elements(By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[3]')
    Cote_A = driver.find_elements(By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[4]')
    Cote_null = driver.find_elements(
        By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[5]'
    )
    Cote_B = driver.find_elements(By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[6]')
    Nb_BM = driver.find_elements(By.XPATH, '//*[@id="tournamentTable"]/tbody/tr/td[7]')
    Année = driver.find_elements(
        By.XPATH, '//*[@id="app"]/div/div[1]/div/main/div[2]/div[6]/div[2]/div[2]/a[1]'
    )[0].text

    df_test = pd.DataFrame(
        columns=[
            "Année",
            "Rencontres",
            "Score",
            "Cote_A",
            "Cote_null",
            "Cote_B",
            "Nb_BM",
        ]
    )
    for i in range(len(Score)):
        df_test = df_test.append(
            {
                "Année": Année,
                "Rencontres": Rencontres[i].text,
                "Score": Score[i].text,
                "Cote_A": Cote_A[i].text,
                "Cote_null": Cote_null[i].text,
                "Cote_B": Cote_B[i].text,
                "Nb_BM": Nb_BM[i].text,
            },
            ignore_index=True,
        )
    return df_test


def orga_DF(df_test: pd.DataFrame) -> pd.DataFrame:
    """Organise la base de données"""
    df_test["Rencontres"].replace("", np.nan, inplace=True)
    df_test.dropna(subset=["Rencontres"], inplace=True)
    df_test[["Equipe_A", "Equipe_B"]] = df_test.Rencontres.str.split(" - ", expand=True)
    df_test[["Score_A", "Score_B"]] = df_test.Score.str.split(":", expand=True)
    date = []
    for i in df_test["Année"]:
        date.append(i.replace("/", "-"))
    df_test["Année"] = date
    df_test = df_test.drop(["Rencontres", "Score"], axis=1)
    df_test["Equipe_A"] = df_test["Equipe_A"].str.replace("\n ", "")
    df_test["Equipe_B"] = df_test["Equipe_B"].str.replace("\n ", "")
    df_test.reindex(
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
    return df_test


def banner(driver: webdriver):
    """Accepte les cookies"""
    try:
        driver.find_elements(By.ID, "onetrust-accept-btn-handler").click()
    except:
        pass


def derniere_page(driver: webdriver) -> int:
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


def page(url: str) -> int:
    """Permet de set up le nombre de page de la saison"""
    option = webdriver.ChromeOptions()
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_argument("--profile-directory=Default")
    driver = webdriver.Chrome(options=option)
    driver.get(url)
    banner(driver)
    page_finale = derniere_page(driver)
    return page_finale


def final(url: str) -> pd.DataFrame:
    """Execute toutes les fonctions créees"""
    driver = setup(url)
    banner(driver)
    time.sleep(3)
    df = recup_DF(driver)
    df = orga_DF(df)
    driver.close()
    return df


def crea_DF_sauf_der_saison() -> pd.DataFrame:
    """Crée le data frame final en ittérant sur l'url sauf la dernière saison"""
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
    for i in range(2004, 2022)[::-1]:
        fin_saison = str(i + 1)[:]
        saison = str(i) + "-" + fin_saison
        page_finale = page(
            "https://www.oddsportal.com/soccer/france/ligue-1-{}/results/#/page/1/".format(
                saison
            )
        )
        for j in range(1, page_finale + 1):
            fin_saison = str(i + 1)[:]
            saison = str(i) + "-" + fin_saison
            df_temporaire = final(
                "https://www.oddsportal.com/soccer/france/ligue-1-{}/results/#/page/{}/".format(
                    saison, j
                )
            )
            df_final = pd.concat([df_final, df_temporaire])
    df_final.reset_index(drop=True, inplace=True)
    data_final = data_final.fillna(value=NA)
    data_final = data_final.dropna()
    data_final["Score_A"] = data_final.Score_A.apply(lambda x: x[0])
    data_final["Score_B"] = data_final.Score_B.apply(lambda x: x[0])
    return df_final


def crea_DF_derniere_saison_test() -> pd.DataFrame:
    """Crée le data frame final en ittérant sur l'url pour la dernière saison"""
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
        page_finale = page(
            "https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/1/"
        )
        for j in range(1, page_finale + 1):
            df_temporaire = final(
                "https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/{}/".format(
                    j
                )
            )
            df_final = pd.concat([df_final, df_temporaire])
    df_final.reset_index(drop=True, inplace=True)
    data_final = data_final.fillna(value=NA)
    data_final.dropna()
    df_final["Score_A"] = df_final["Score_A"].astype(str)
    df_final["Score_B"] = df_final["Score_B"].astype(str)
    df_final["Score_A"] = df_final.Score_A.apply(lambda x: x[0])
    df_final["Score_B"] = df_final.Score_B.apply(lambda x: x[0])
    return df_final


def execute_scraping() -> csv:
    """Dernière fonction permmetant d'excuter le scraping et crée le csv final"""
    df1 = crea_DF_sauf_der_saison()
    df2 = crea_DF_derniere_saison_test()
    df_finale = pd.concat(df2, df1)
    return df_finale.to_csv(
        r"C:\Users\pierr\Desktop\M2_Economie\Machine_Learning\Projet_machine_learning\Scraping\DF_finale.csv",
        index=False,
        header=True,
        sep=";",
        encoding="utf-8-sig",
    )
