"""
Cette librairie permet de générer de nouvelles colonnes à partir de calculs statistiques.
C'est un regroupement et un embellissement des deux bases de données scrapées.

Après soutenance :

- Ajout de l'incateur Elo_rating
- Ajout des 3 composantes principales créées à partir d'une analyse par 
composantes principales qui regroupe plus de 96% de la variance expliquée.
Elle remplace les variables utilisées auparavant comme préditeurs :
Diff_Victoire, Diff_Nul, Diff_Défaite, Diff_points, Diff_but, Diff_encaisse, Diff_Moyenne_cote
"""

from pandas import NA
import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA
from joblib import dump, load

warnings.filterwarnings("ignore")

####### CSV issu du scrapping #######

extract1 = pd.read_csv(
    "/Users/alexandrelaurent/Desktop/Version_Final_ML/Extraction1.csv",
    sep=";",
    encoding="utf-8-sig",
)

extract2 = pd.read_csv(
    "/Users/alexandrelaurent/Desktop/Version_Final_ML/Extraction2.csv",
    sep=";",
    encoding="utf-8-sig",
)
extract = pd.concat([extract2, extract1]).reset_index()

####### Classe gestion des données #######


class Data_match:
    """
    Classe qui gère l'approvisionnement, la recherche de statistiques,
    la création automatisé des variables explicatives destinées aux modèle de ML
    et la visualisation des données.
    """

    def __init__(self):

        self.equipe_D = extract["Equipe_A"].rename("Equipe_D")
        self.equipe_E = extract["Equipe_B"].rename("Equipe_E")
        self.cote_D = extract["Cote_A"].rename("Cote_D")
        self.cote_N = extract["Cote_null"].rename("Cote_N")
        self.cote_E = extract["Cote_B"].rename("Cote_E")
        self.score_D = extract["Score_A"].rename("Score_D")
        self.score_E = extract["Score_B"].rename("Score_E")
        self.saison = extract["Année"].rename("Saison")

    def Show_extract(self) -> pd.DataFrame:
        """
        Visualiser la base de données extraites du scrapping.
        """
        Data = pd.DataFrame(
            [
                self.equipe_D,
                self.equipe_E,
                self.cote_D,
                self.cote_N,
                self.cote_E,
                self.score_D,
                self.score_E,
                self.saison,
            ]
        ).T
        Data = Data.drop(Data[(Data["Cote_D"] == "-") | (Data["Cote_E"] == "-")].index)
        Data = Data.astype(
            {"Score_D": int, "Score_E": int, "Cote_D": float, "Cote_E": float}
        )
        return Data.reset_index()

    def N_last_match(self, Taille: int, equipe: str) -> pd.DataFrame:
        """
        Retourne un dataframe des N derniers matchs de l'équipe sélectionné.
        """
        Data = Data_match().Show_extract()
        i = _index_N_dernier_macth(Taille, equipe)
        return Data.loc[
            i,
        ]

    def Statistiques(self, Taille: int, equipe1: str, equipe2: str) -> pd.DataFrame:

        info1, info2 = _Statistiques_n_dernier_match(
            Taille, equipe1
        ), _Statistiques_n_dernier_match(Taille, equipe2)
        pts1, pts2 = _Classement_Now(equipe1), _Classement_Now(equipe2)
        Elo_D, Elo_E = _Elo_Now(equipe1), _Elo_Now(equipe2)

        Stats = pd.DataFrame(
            [
                [str(info1[0]), str(info2[0])],
                [info1[1], info2[1]],
                [info1[2], info2[2]],
                [info1[3], info2[3]],
                [info1[4], info2[4]],
                [info1[5], info2[5]],
                [pts1, pts2],
                [Elo_D, Elo_E],
            ],
            index=[
                "Victoire",
                "Nul",
                "Defaite",
                "But",
                "Encaisse",
                "Moyenne_cote",
                "Points",
                "Elo_rating",
            ],
            columns=[equipe1, equipe2],
        )
        return Stats

    def Compare_perf_equipe(
        self, Taille: int, equipe_D: str, equipe_E: str
    ) -> pd.DataFrame:
        """
        Retourne un dataframe présentant les statistiques sur les N dernier matchs
        joués par les deux équipes sélectionnées.
        """
        info_D, info_E = _Statistiques_n_dernier_match(
            Taille, equipe_D
        ), _Statistiques_n_dernier_match(Taille, equipe_E)
        pts1, pts2 = _Classement_Now(equipe_D), _Classement_Now(equipe_E)
        Elo_D, Elo_E = _Elo_Now(equipe_D), _Elo_Now(equipe_E)

        Stats = pd.DataFrame(
            {
                "Diff_Victoire": [info_D[0] - info_E[0]],
                "Diff_Nul": [info_D[1] - info_E[1]],
                "Diff_Defaite": [info_D[2] - info_E[2]],
                "Diff_But": [info_D[3] - info_E[3]],
                "Diff_Encaisse": [info_D[4] - info_E[4]],
                "Diff_Moyenne_cote": [info_D[5] - info_E[5]],
                "Diff_Points": [pts1 - pts2],
                "Diff_Elo_rating": [Elo_D - Elo_E],
            },
            index=[str(equipe_D) + "-" + str(equipe_E)],
        )

        return Stats

    def Dt_target(self) -> pd.DataFrame:
        Data = _Target()
        try:
            del Data["index"]
        except:
            pass
        return Data

    def _Add_variables_DT(self, Taille: int) -> pd.DataFrame:
        Data = _Ajout_Variables(Taille)
        del Data["index"]
        return Data

    def Final_DataFrame(self, Taille: int) -> pd.DataFrame:
        Classement = _Classement_equipe(Taille)
        del Classement["level_0"]
        Classement["PCA1"], Classement["PCA2"], Classement["PCA3"] = (
            _ACP(Classement)[0],
            _ACP(Classement)[1],
            _ACP(Classement)[2],
        )
        Classement.to_csv("Data.csv")
        return Classement


####### Fonctions utilent à la construction #######


def _Target() -> pd.DataFrame:
    """
    Permet d'ajouter la valeur à prédire au dataframe
    """
    Data = Data_match().Show_extract()
    Winner_D = []
    Winner_E = []

    for but_home, but_exter in zip(Data["Score_D"], Data["Score_E"]):
        if but_home > but_exter:
            Winner_D.append(1)
            Winner_E.append(0)
        elif but_home < but_exter:
            Winner_D.append(0)
            Winner_E.append(1)
        else:
            Winner_D.append(0)
            Winner_E.append(0)

    Data["Winner_D"] = Winner_D
    Data["Winner_E"] = Winner_E

    return Data


def _index_N_dernier_macth(Taille: int, equipe: str) -> list[int]:
    """
    Permet de récupérer les indices des N derniers match d'une équipe
    """
    Data = Data_match().Show_extract()
    Equipe1 = equipe

    idx1 = np.where((Data["Equipe_D"] == Equipe1) | (Data["Equipe_E"] == Equipe1))[
        0
    ].tolist()
    Nindex1 = idx1[0:Taille]

    return Nindex1


####### Variables Statistiques sur N derniers matchs #######


def _Statistiques_n_dernier_match(Taille: int, equipe: str) -> list[float]:
    """
    Permet de calculer les statistiques d'avant match sur les N dernier match d'une équipe sélectionné.
    """

    victoire, nul, defaite, but, encaisse, moyenne_cote = 0, 0, 0, 0, 0, 0
    Data = Data_match().Show_extract()
    index = _index_N_dernier_macth(Taille, equipe)

    if len(index) == Taille:
        for i in index:

            if (
                Data["Equipe_D"].loc[i] == equipe
                and Data["Score_D"].loc[i] > Data["Score_E"].loc[i]
            ):
                victoire += 1
                but += Data["Score_D"].loc[i]
                encaisse += Data["Score_E"].loc[i]
                moyenne_cote += Data["Cote_D"].loc[i] / Taille

            elif (
                Data["Equipe_E"].loc[i] == equipe
                and Data["Score_E"].loc[i] > Data["Score_D"].loc[i]
            ):
                victoire += 1
                but += Data["Score_E"].loc[i]
                encaisse += Data["Score_D"].loc[i]
                moyenne_cote += Data["Cote_E"].loc[i] / Taille

            elif (
                Data["Equipe_D"].loc[i] == equipe
                and Data["Score_D"].loc[i] == Data["Score_E"].loc[i]
            ):
                nul += 1
                but += Data["Score_D"].loc[i]
                encaisse += Data["Score_E"].loc[i]
                moyenne_cote += Data["Cote_D"].loc[i] / Taille

            elif (
                Data["Equipe_E"].loc[i] == equipe
                and Data["Score_E"].loc[i] == Data["Score_D"].loc[i]
            ):
                nul += 1
                but += Data["Score_E"].loc[i]
                encaisse += Data["Score_D"].loc[i]
                moyenne_cote += Data["Cote_E"].loc[i] / Taille

            elif (
                Data["Equipe_D"].loc[i] == equipe
                and Data["Score_D"].loc[i] < Data["Score_E"].loc[i]
            ):
                defaite += 1
                but += Data["Score_D"].loc[i]
                encaisse += Data["Score_E"].loc[i]
                moyenne_cote += Data["Cote_D"].loc[i] / Taille

            elif (
                Data["Equipe_E"].loc[i] == equipe
                and Data["Score_E"].loc[i] < Data["Score_D"].loc[i]
            ):
                defaite += 1
                but += Data["Score_E"].loc[i]
                encaisse += Data["Score_D"].loc[i]
                moyenne_cote += Data["Cote_E"].loc[i] / Taille

            else:
                victoire, nul, defaite, encaisse, moyenne_cote, but = (
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                )
    else:
        victoire, nul, defaite, encaisse, moyenne_cote, but = NA, NA, NA, NA, NA, NA

    return victoire, nul, defaite, but, encaisse, moyenne_cote


####### Création des vaiables explicatives #######


def _Variables_explicatives(Taille: int, equipe: str) -> list[list[float]]:
    """
    Permet de récupérer les information sur les N derniers matchs des équipes pour tout le dataframe.
    Deux choix possibles : equipe = 'Equipe_D' ou equipe = 'Equipe_E'
    """

    Data = Data_match().Show_extract()
    Victoire, Nul, Defaite, Moyenne_cote, But, Encaisse = [], [], [], [], [], []

    for i in range(0, len(Data)):
        Index = i
        victoire, nul, defaite, but, encaisse, moyenne_cote = 0, 0, 0, 0, 0, 0
        Equipe1 = Data[equipe].loc[Index]
        idx1 = np.where((Data["Equipe_D"] == Equipe1) | (Data["Equipe_E"] == Equipe1))[
            0
        ].tolist()
        Five1 = idx1[idx1.index(Index) + 1 : idx1.index(Index) + 1 + Taille]

        if len(Five1) == Taille:
            for i in Five1:

                if (
                    Data["Equipe_D"].loc[i] == Equipe1
                    and Data["Score_D"].loc[i] > Data["Score_E"].loc[i]
                ):
                    victoire += 1
                    but += Data["Score_D"].loc[i]
                    encaisse += Data["Score_E"].loc[i]
                    moyenne_cote += Data["Cote_D"].loc[i] / Taille

                elif (
                    Data["Equipe_E"].loc[i] == Equipe1
                    and Data["Score_E"].loc[i] > Data["Score_D"].loc[i]
                ):
                    victoire += 1
                    but += Data["Score_E"].loc[i]
                    encaisse += Data["Score_D"].loc[i]
                    moyenne_cote += Data["Cote_E"].loc[i] / Taille

                elif (
                    Data["Equipe_D"].loc[i] == Equipe1
                    and Data["Score_D"].loc[i] == Data["Score_E"].loc[i]
                ):
                    nul += 1
                    but += Data["Score_D"].loc[i]
                    encaisse += Data["Score_E"].loc[i]
                    moyenne_cote += Data["Cote_D"].loc[i] / Taille

                elif (
                    Data["Equipe_E"].loc[i] == Equipe1
                    and Data["Score_E"].loc[i] == Data["Score_D"].loc[i]
                ):
                    nul += 1
                    but += Data["Score_E"].loc[i]
                    encaisse += Data["Score_D"].loc[i]
                    moyenne_cote += Data["Cote_E"].loc[i] / Taille

                elif (
                    Data["Equipe_D"].loc[i] == Equipe1
                    and Data["Score_D"].loc[i] < Data["Score_E"].loc[i]
                ):
                    defaite += 1
                    but += Data["Score_D"].loc[i]
                    encaisse += Data["Score_E"].loc[i]
                    moyenne_cote += Data["Cote_D"].loc[i] / Taille

                elif (
                    Data["Equipe_E"].loc[i] == Equipe1
                    and Data["Score_E"].loc[i] < Data["Score_D"].loc[i]
                ):
                    defaite += 1
                    but += Data["Score_E"].loc[i]
                    encaisse += Data["Score_D"].loc[i]
                    moyenne_cote += Data["Cote_E"].loc[i] / Taille

                else:
                    victoire, nul, defaite, but, encaisse, moyenne_cote = (
                        NA,
                        NA,
                        NA,
                        NA,
                        NA,
                        NA,
                    )

            Victoire.append(victoire), Nul.append(nul), Defaite.append(defaite),
            Moyenne_cote.append(moyenne_cote), But.append(but), Encaisse.append(
                encaisse
            )

        else:
            Victoire.append(NA), Nul.append(NA), Defaite.append(NA),
            Moyenne_cote.append(NA), But.append(NA), Encaisse.append(NA)

    return Victoire, Nul, Defaite, Moyenne_cote, But, Encaisse


def _Ajout_Variables(Taille: int) -> pd.DataFrame:
    """
    Ajout des prédicteurs dans la base de données
    """

    Data = Data_match().Dt_target()
    Elo = _Elo_rating()
    Data["Elo_D_Before"] = Elo[1]
    Data["Elo_D"] = Elo[0]
    Data["Elo_E_Before"] = Elo[3]
    Data["Elo_E"] = Elo[2]

    Domicile = _Variables_explicatives(Taille, "Equipe_D")
    Exterieur = _Variables_explicatives(Taille, "Equipe_E")

    Diff_Victoire = [D - E for D, E in zip(Domicile[0], Exterieur[0])]
    Diff_Nul = [D - E for D, E in zip(Domicile[1], Exterieur[1])]
    Diff_Defaite = [D - E for D, E in zip(Domicile[2], Exterieur[2])]
    Diff_Moyenne_cote = [D - E for D, E in zip(Domicile[3], Exterieur[3])]
    Diff_But = [D - E for D, E in zip(Domicile[4], Exterieur[4])]
    Diff_Encaisse = [D - E for D, E in zip(Domicile[5], Exterieur[5])]
    Diff_Elo_Before = [D - E for D, E in zip(Elo[1], Elo[3])]

    Data["Diff_Victoire"] = Diff_Victoire
    Data["Diff_Nul"] = Diff_Nul
    Data["Diff_Defaite"] = Diff_Defaite
    Data["Diff_But"] = Diff_But
    Data["Diff_Encaisse"] = Diff_Encaisse
    Data["Diff_Moyenne_cote"] = Diff_Moyenne_cote
    Data["Diff_Elo_Before"] = Diff_Elo_Before

    return Data.dropna().reset_index()


def _Calcul_points(equipe: str, Dataframe) -> list[int]:
    """
    Récupère les indexes où se trouve l'équipe dans une liste
    qu'elle soit à l'extèrieur ou à domicile.

    Pour chaque index de la liste on récupère toutes les rencontres précédentes,
    on les parcours en comptant le nombre de points.
    """

    Data_test = Dataframe
    Points = []
    for i in range(0, len(Data_test) - 1):
        Index = i
        pts = 0
        Equipe1 = Data_test[equipe].loc[Index]
        idx1 = np.where(
            (Data_test["Equipe_D"] == Equipe1) | (Data_test["Equipe_E"] == Equipe1)
        )[0].tolist()
        rencontres = idx1[idx1.index(Index) + 1 : len(idx1)]

        for i in rencontres:

            if (
                Data_test["Equipe_D"].loc[i] == Equipe1
                and Data_test["Score_D"].loc[i] > Data_test["Score_E"].loc[i]
            ):
                pts += 3

            if (
                Data_test["Equipe_E"].loc[i] == Equipe1
                and Data_test["Score_E"].loc[i] > Data_test["Score_D"].loc[i]
            ):
                pts += 3

            if (
                Data_test["Equipe_D"].loc[i] == Equipe1
                and Data_test["Score_D"].loc[i] == Data_test["Score_E"].loc[i]
            ):
                pts += 1

            if (
                Data_test["Equipe_E"].loc[i] == Equipe1
                and Data_test["Score_E"].loc[i] == Data_test["Score_D"].loc[i]
            ):
                pts += 1

            if (
                Data_test["Equipe_D"].loc[i] == Equipe1
                and Data_test["Score_D"].loc[i] < Data_test["Score_E"].loc[i]
            ):
                pts += 0

            if (
                Data_test["Equipe_E"].loc[i] == Equipe1
                and Data_test["Score_E"].loc[i] < Data_test["Score_D"].loc[i]
            ):
                pts += 0

        Points.append(pts)
    return Points


def _Classement_equipe(Taille) -> pd.DataFrame:
    """
    Utilise la fonction _Calcul_points()

    Initialisation : Calcul les points d'avant matchpour chaque rencontre sur la saison en cours
    Crée un premier Dataframe pandas de la saison en cours avec classement

    Traitement successif : Répète l'opération dans une boucle saison par saison
    et concat la saison faites avec la/les saisons précédentes.
    """

    Data = Data_match()._Add_variables_DT(Taille)  # Base entière

    # Initialisation

    Data2 = Data.loc[Data["Saison"] == Data.Saison[0]].reset_index()  # Base alternative
    Points_Domicile = _Calcul_points("Equipe_D", Data2)  # Récupère les points_D
    Points_Domicile.append(0)
    Points_Exterieur = _Calcul_points("Equipe_E", Data2)  # Récupère les points_E
    Points_Exterieur.append(0)
    Data2["Points_a"] = Points_Domicile  # Met dans data2
    Data2["Points_b"] = Points_Exterieur  # Met dans data2
    Data2["Diff_Points"] = [
        PD - PE for PD, PE in zip(Points_Domicile, Points_Exterieur)
    ]
    Data = Data.loc[Data["Saison"] != Data.Saison[0]]

    # Traitement pour chaque saison et ajout successif

    for s in Data["Saison"].unique():
        Data_saison = Data.loc[
            Data["Saison"] == s
        ].reset_index()  # Prend la saison et reset index
        Points_Domicile = _Calcul_points("Equipe_D", Data_saison)
        Points_Domicile.append(0)
        Points_Exterieur = _Calcul_points("Equipe_E", Data_saison)
        Points_Exterieur.append(0)
        Data_saison["Points_a"] = Points_Domicile
        Data_saison["Points_b"] = Points_Exterieur
        Data_saison["Diff_Points"] = [
            PD - PE for PD, PE in zip(Points_Domicile, Points_Exterieur)
        ]
        Data2 = pd.concat([Data2, Data_saison])

    Data2 = Data2.reset_index()
    del Data2["index"]

    return Data2


def _Classement_Now(equipe: str) -> int:
    """
    Donne le nom de l'équipe et renvoit le classement actuel
    d'une équipe en points selon les règles mis en place par la FFF :

    Gagne : + 3 pts
    Nul : + 1 pts
    Perd : 0 pts
    """

    data = Data_match().Show_extract()
    match_saison = data.loc[
        ((data["Equipe_D"] == equipe) | (data["Equipe_E"] == equipe))
        & (data["Saison"] == data.Saison[0])
    ].reset_index()

    if match_saison["Equipe_D"][0] == equipe:
        pts = _Calcul_points("Equipe_D", match_saison)[0]
        if match_saison["Score_D"][0] > match_saison["Score_E"][0]:
            pts += 3
        if match_saison["Score_D"][0] == match_saison["Score_E"][0]:
            pts += 1
        else:
            pass
    else:
        pts = _Calcul_points("Equipe_E", match_saison)[0]
        if match_saison["Score_E"][0] > match_saison["Score_D"][0]:
            pts += 3
        if match_saison["Score_E"][0] == match_saison["Score_D"][0]:
            pts += 1
        else:
            pass

    return pts


def _Elo_rating() -> list[list[float]]:

    data = Data_match().Dt_target()
    data_inverse = data.iloc[::-1].reset_index()
    del data_inverse["index"]
    Elo_D, Elo_E = [], []
    repere = {}
    elo_D = 0
    Elo_D_Before = []
    Elo_E_Before = []
    elo_E = 0
    gagne = 1
    nul = 0.5
    perd = 0

    for i in data_inverse["Equipe_D"].unique():
        repere.update({i: 1500})

    for i in range(0, len(data_inverse)):

        if i != 0:
            if data_inverse["Saison"].loc[i] != data_inverse["Saison"].loc[i - 1]:
                for equipe in data["Equipe_D"].unique():
                    repere.update({equipe: repere[equipe] * 0.90 + 0.10 * 1500})

        Equipe1 = data_inverse["Equipe_D"].loc[i]
        Equipe2 = data_inverse["Equipe_E"].loc[i]

        elo_D = repere.get(Equipe1)
        elo_E = repere.get(Equipe2)

        Elo_D_Before.append(elo_D)
        Elo_E_Before.append(elo_E)

        EA_D = 1 / (1 + 10 ** ((repere.get(Equipe2) - repere.get(Equipe1)) / 400))
        EA_E = 1 / (1 + 10 ** ((repere.get(Equipe1) - repere.get(Equipe2)) / 400))
        k = 20

        if data_inverse["Score_D"].loc[i] > data_inverse["Score_E"].loc[i]:

            elo_D += k * (gagne - EA_D)
            elo_E += k * (perd - EA_E)

        elif data_inverse["Score_D"].loc[i] == data_inverse["Score_E"].loc[i]:

            elo_D += k * (nul - EA_D)
            elo_E += k * (nul - EA_E)

        else:

            elo_D += k * (perd - EA_D)
            elo_E += k * (gagne - EA_E)

        Elo_D.append(elo_D)
        Elo_E.append(elo_E)

        repere.update({Equipe1: elo_D}), repere.update({Equipe2: elo_E})
    return (
        list(reversed(Elo_D)),
        list(reversed(Elo_D_Before)),
        list(reversed(Elo_E)),
        list(reversed(Elo_E_Before)),
    )


def _ACP(Data) -> list[list[float]]:
    pca = PCA(n_components=3, whiten=True)
    VA = pca.fit_transform(
        Data[
            [
                "Diff_Victoire",
                "Diff_Nul",
                "Diff_Defaite",
                "Diff_But",
                "Diff_Encaisse",
                "Diff_Moyenne_cote",
                "Diff_Points",
            ]
        ]
    )
    dump(pca, "PCA.joblib")
    pcA1, pcA2, pcA3 = [], [], []
    for i in VA:
        pcA1.append(i[0]), pcA2.append(i[1]), pcA3.append(i[2])
    return pcA1, pcA2, pcA3


def _Elo_Now(equipe: str) -> float:

    data = pd.read_csv("/Users/alexandrelaurent/Desktop/Version_Final_ML/Data.csv")

    match_saison = data.loc[
        ((data["Equipe_D"] == equipe) | (data["Equipe_E"] == equipe))
    ].reset_index()

    if match_saison["Equipe_D"][0] == equipe:
        Elo = round(match_saison["Elo_D"][0], 2)

    elif match_saison["Equipe_E"][0] == equipe:
        Elo = round(match_saison["Elo_E"][0], 2)
    return Elo
