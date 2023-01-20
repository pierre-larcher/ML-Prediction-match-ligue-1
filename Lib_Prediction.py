"""
Cette librairie présente une classe nommée 'Prediction' qui permet de :

- Faire différent modèles directement à partir de la classe
- Enregistrer directement ces modèles dans le dossier du projet en .joblib.
- Recharger un modèle, visualiser ces performances ('score') ou ('visualisation graphique réalité/Prédiction')
"""

from importlib import reload
import Lib_data as LD

reload(LD)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from joblib import dump, load
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from IPython.display import HTML, display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import ipywidgets as widgets
from pandas import NA


class Prediction:
    def __init__(self, target: str, predictor: object, Dom_Ext: str):

        self.modele = {
            "rfr_gs": "Random_Forest",
            "pln_gs": "Reseau_Neuronnes",
            "lr_gs": "Logistic_Regression",
            "knn_gs": "K_Nearest_Neighbors",
            "xgb_gs": "XgB_Boost",
            "ada_gs": "Ada_Boost",
            "svc_gs": "Support Vecteur Machine",
        }
        self.equipe = Dom_Ext
        self.target = target
        self.predictor = predictor
        (
            self.predictor_train,
            self.predictor_test,
            self.target_train,
            self.target_test,
        ) = train_test_split(self.predictor, self.target)
        oversample = RandomOverSampler()
        self.predictor_over, self.target_over = oversample.fit_resample(
            self.predictor_train, self.target_train
        )

    def Reseau_neuronne(self) -> list[tuple]:
        """
        Entraine un réseau de neuronnes ,fit sur l'ensemble de test.
        Enregistre le modèle dans un fichier :
        'Reseau_Neuronnes' + 'domicile ou exterieur' .joblib

        Retourne le meilleur hyperparamétrage et la précision associée
        """

        pln = Pipeline(
            [
                ("mise_echelle", MinMaxScaler()),
                ("neurones", MLPClassifier()),
            ]
        )

        pln_gs = GridSearchCV(
            pln,
            {
                "neurones__hidden_layer_sizes": [(10,), (100,), (200,), (100, 50)],
                "neurones__activation": ["identity", "logistic", "tanh"],
                "neurones__alpha": 10.0 ** -np.arange(1, 7),
                "neurones__learning_rate": ["constant", "invscaling", "adaptive"],
            },
            scoring="precision",
        )

        pln_gs.fit(self.predictor_train, self.target_train)
        dump(pln_gs, str(self.modele["pln_gs"]) + "_" + str(self.equipe) + ".joblib")
        return pln_gs.best_params_, pln_gs.best_score_

    def Random_Forest(self) -> list[tuple]:

        rfr = RandomForestClassifier()
        rfr_gs = GridSearchCV(
            rfr,
            {
                "n_estimators": (2, 4, 8, 16, 32, 64, 128, 256, 512),
                "criterion": ("gini", "entropy", "log_loss"),
                "max_leaf_nodes": range(1, 20),
            },
            scoring="precision",
        )

        rfr_gs.fit(self.predictor_train, self.target_train)
        dump(rfr_gs, str(self.modele["rfr_gs"]) + "_" + str(self.equipe) + ".joblib")
        return rfr_gs.best_params_, rfr_gs.best_score_

    def Support_vector_machine(self) -> list[tuple]:

        svc = Pipeline(
            [
                ("standardscaler", StandardScaler()),
                ("support", SVC()),
            ]
        )
        svc_gs = GridSearchCV(
            svc,
            {
                "support__C": [0, 0.001, 0.05, 0.1, 1.0, 10],
                "support__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                "support__probability": [True],
            },
            scoring="precision",
        )

        svc_gs.fit(self.predictor_over, self.target_over)
        dump(svc_gs, str(self.modele["svc_gs"]) + "_" + str(self.equipe) + ".joblib")
        return svc_gs.best_params_, svc_gs.best_score_

    def K_Nearest_Neighbors(self) -> list[tuple]:
        """
        Entraine une classification avec les K-plus proche voisin, fit sur l'ensemble de test.
        Enregistre le modèle dans un fichier :
        'K_Nearest_Neighbors' + 'domicile ou exterieur' .joblib

        Retourne le meilleur hyperparamétrage et la précision associée
        """
        knn = KNeighborsClassifier()

        knn_gs = GridSearchCV(
            knn,
            {
                "n_neighbors": range(5, 100),
                "weights": ("uniform", "distance"),
                "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
                "p": (1, 2),
            },
            scoring="precision",
        )

        knn_gs.fit(self.predictor_train, self.target_train)
        dump(knn_gs, str(self.modele["knn_gs"]) + "_" + str(self.equipe) + ".joblib")
        return knn_gs.best_params_, knn_gs.best_score_

    def XgB_Boost(self) -> list[tuple]:
        """
        Entraine une classification avec XgB_Boost, fit sur l'ensemble de test.
        Enregistre le modèle dans un fichier :
        'XgB_Boost' + 'domicile ou exterieur' .joblib

        Retourne le meilleur hyperparamétrage et la précision associée
        """

        self.predictor_test = self.predictor_test.astype(float)

        xgb = XGBClassifier()

        xgb_gs = GridSearchCV(
            xgb,
            {
                "learning_rate": [0, 0.5, 1, 1.5, 2],
                "n_estimators": [2, 4, 8, 16, 31, 64, 128, 256],
                "min_child_weight": [2, 4, 8],
                "gamma": [0, 1, 2],
            },
            scoring="precision",
        )

        xgb_gs.fit(self.predictor_train.astype(float), self.target_train)
        dump(xgb_gs, str(self.modele["xgb_gs"]) + "_" + str(self.equipe) + ".joblib")
        return xgb_gs.best_params_, xgb_gs.best_score_

    def Ada_boost(self) -> list[tuple]:
        ada = GradientBoostingClassifier()

        ada_gs = GridSearchCV(
            ada,
            {
                "loss": ("log_loss", "deviance", "exponential"),
                "learning_rate": (0, 2, 4, 16, 32),
                "n_estimators": (2, 4, 8, 16, 32),
                "criterion": ("friedman_mse", "squared_error", "mse"),
                "max_leaf_nodes": range(1, 10),
            },
            scoring="precision",
        )

        ada_gs.fit(self.predictor_train, self.target_train)

        ada_gs.best_params_, ada_gs.best_score_
        dump(ada_gs, str(self.modele["ada_gs"]) + "_" + str(self.equipe) + ".joblib")
        return ada_gs.best_params_, ada_gs.best_score_

    def Logistic_Regression(self) -> list[tuple]:
        """
        Entraine une classification Logistique, fit sur l'ensemble de test.
        Enregistre le modèle dans un fichier :
        'Logistic_Regression' + 'domicile ou exterieur' .joblib

        Retourne le meilleur hyperparamétrage et la précision associée
        """

        lr = LogisticRegression()

        lr_gs = GridSearchCV(
            lr,
            {
                "penalty": ("l2", "none", "l1", "elasticnet"),
                "C": (0, 0.001, 0.05, 0.1, 1.0, 10),
                "tol": (0, 0.0000001, 0.000001, 0.00001),
                "class_weight": ("dict", "balanced"),
                "solver": (
                    "newton-cg",
                    "lbfgs",
                    "liblinear",
                    "sag",
                    "saga",
                    "newton-cholesky",
                ),
                "fit_intercept": (True, False),
            },
            scoring="precision",
        )

        lr_gs.fit(self.predictor_train, self.target_train)

        lr_gs.best_params_, lr_gs.best_score_
        dump(lr_gs, str(self.modele["lr_gs"]) + "_" + str(self.equipe) + ".joblib")
        return lr_gs.best_params_, lr_gs.best_score_

    def Load_Modele(self, modele: str):
        """
        On lui donne le modèle à charger.
        Retourne le modèle
        """

        return load(str(self.modele[modele]) + "_" + str(self.equipe) + ".joblib")

    def Performances(self, modele) -> plt:
        """
        Donne le modèle et retourne la matrice de confusion et tout les indicateur de scoring :
        - Précision
        - Spécificité
        - Sensibilité
        - AUC
        - Précision positif
        - Précision négatif
        Sous forme de Dataframe
        """

        Mc = confusion_matrix(self.target_test, modele.predict(self.predictor_test))

        ax = plt.subplot()
        sns.heatmap(Mc, annot=True, ax=ax, fmt="g")
        # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel("Prediction", fontsize=20)
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticklabels(["Autre", str(self.equipe)], fontsize=15)
        ax.xaxis.tick_top()

        ax.set_ylabel("Réalité", fontsize=20)
        ax.yaxis.set_ticklabels(["Autre", str(self.equipe)], fontsize=15)

        predict_cv = modele.predict(self.predictor_test)
        roc = roc_auc_score(self.target_test, predict_cv)
        acc = accuracy_score(self.target_test, predict_cv)

        VN = Mc[0, 0]
        VP = Mc[1, 1]
        FP = Mc[0, 1]
        FN = Mc[1, 0]

        # Capacité à repérer les vrais négatifs
        Spécificité = VN / (VN + FP)
        # Capacité à repérer les vrais positifs (rappel)
        Sensibilité = VP / (VP + FN)
        # Précision du modèle
        Précision = (VP + VN) / (VP + VN + FP + FN)
        # Présion des positif
        PréciP = VP / (VP + FP)
        # Précision des négatif
        PréciN = VN / (VN + FN)

        TabP = pd.DataFrame(
            [
                [
                    Précision,
                    Sensibilité,
                    Spécificité,
                    PréciP,
                    PréciN,
                    roc,
                    acc,
                    Spécificité + Sensibilité,
                ]
            ],
            index=["Score"],
            columns=[
                "Précision",
                "Sensibilité",
                "Spécificité",
                "Précision Positif",
                "Précision Négatif",
                "AUC",
                "Accuracy",
                "Spé+Sen",
            ],
        )
        plt.show(), display(round(TabP, 2))

    def Tab_Realite_Pred(self, modele) -> pd.DataFrame:
        """
        Choisir le modèle et renvois un Dataframe avec la prédiction faite
        sur l'ensemble de test, la réalité (target), le prédict proba,
        l'espérance de gain et la décision.
        """

        if self.equipe == "exterieur":
            Cote = "Cote_E"
        elif self.equipe == "domicile":
            Cote = "Cote_D"
        else:
            raise ValueError(
                "Pour quelle équipe voulez-vous prédire ? 'exterieur' ou 'domicile'"
            )

        y_te = self.target_test
        X_te = self.predictor_test

        predictions = y_te

        Prediction = pd.DataFrame(predictions).reset_index()
        Variables = pd.DataFrame(X_te).reset_index()
        Proba = pd.DataFrame(modele.predict_proba(X_te))

        (
            Prediction["Cote" + str(self.equipe)],
            Prediction["Prediction"],
            Prediction["Autres"],
            Prediction[str(self.equipe)],
        ) = (Variables[Cote], modele.predict(X_te), Proba[0], Proba[1])
        del Prediction["index"]
        Prediction["Esperance"] = (Prediction[str(self.equipe)]) * (
            (Prediction["Cote" + str(self.equipe)]) - 1
        ) - (1 - Prediction[str(self.equipe)])

        Decision = []
        for E, P in zip(Prediction["Esperance"], Prediction["Prediction"]):
            if E > 0 and P == 1:
                Decision.append(1)
            else:
                Decision.append(0)
        Prediction["Decision"] = Decision

        return Prediction

    def Indicateurs_Pred(self, mini, maxi, modele) -> pd.DataFrame:
        """
        Retourne 3 indicateurs pour la prise de décision 'parier ou ne pas parier'
        sur un intervalle de prédiction (min-max)

        Evalue la performance du modèle sur l'ensemble de test dans cette intervalle :
        - Espérance de gain
        - Rentabilité
        - Taux de bonne prédiction
        """
        tab = self.Tab_Realite_Pred(modele)
        data = tab.loc[
            (tab[str(self.equipe)] > mini)
            & (tab[str(self.equipe)] < maxi)
            & (tab["Prediction"] == 1)
        ]
        try:
            Esperance_gain = (
                str(round((sum(data["Esperance"]) / len(data)) * 100, 2)) + "%"
            )
        except:
            Esperance_gain = "X"

        Nb_pred, Bonne_pred, Mauvaise_pred, Investissement, Profit = 0, 0, 0, 0, 0
        for Pred, Real, Cote in zip(
            data["Prediction"],
            data[self.target.columns[0]],
            data["Cote" + str(self.equipe)],
        ):
            if Pred == 1 and Real == 1:

                Profit += Cote - 1
                Bonne_pred += 1

            elif Pred == 1 and Real == 0:
                Profit -= 1
                Mauvaise_pred += 1
            else:
                pass
            Nb_pred += 1
            Investissement += 1

        try:
            Rentabilite = str(round((Profit / Investissement) * 100, 2)) + "%"
        except:
            Rentabilite = "X"

        try:
            perf = str(round((Bonne_pred / Nb_pred) * 100, 2)) + "%"
        except:
            perf = "X"

        return Esperance_gain, Rentabilite, perf

    def Histogramme_pred(self, modele) -> widgets:
        def Visualisation(mini, maxi) -> plt:
            """
            Permet de visualiser deux histogrammes (bien prédit - mal prédit)
            superposés avec en ordonnée les fréquences (nombre de matchs)
            et en abscisse les intervalles de proba sortie de prédict_proba de sklearn.
            Retourne également des indicateurs de performances sous forme de Dataframe pandas
            """

            Real_Pred = self.Tab_Realite_Pred(modele)
            Nb_pred, Bonne_pred, Mauvaise_pred, Investissement, Profit = 0, 0, 0, 0, 0
            R_P_min_max = Real_Pred.loc[
                (Real_Pred[str(self.equipe)] > mini)
                & (Real_Pred[str(self.equipe)] < maxi)
                & (Real_Pred["Prediction"] == 1)
            ]

            for Real, Cote, Pred in zip(
                R_P_min_max[self.target.columns[0]],
                R_P_min_max["Cote" + str(self.equipe)],
                R_P_min_max["Prediction"],
            ):

                if Pred == 1 and Real == 1:
                    Profit += Cote - 1
                    Bonne_pred += 1

                elif Pred == 1 and Real == 0:
                    Profit -= 1
                    Mauvaise_pred += 1
                else:
                    pass
                Nb_pred += 1
                Investissement += 1

            Stats = pd.DataFrame(
                {
                    "Nombre de match": [Nb_pred],
                    "Bien prédit": [Bonne_pred],
                    "Mal prédit": [Mauvaise_pred],
                    "Retour sur investissement": [
                        "{} %".format(round((Profit / Investissement) * 100, 2))
                    ],
                    "Espérance de gain": [
                        "{} %".format(
                            round(
                                (
                                    0.5
                                    * (maxi + mini)
                                    * (
                                        (
                                            sum(R_P_min_max["Cote" + str(self.equipe)])
                                            / len(R_P_min_max)
                                        )
                                        - 1
                                    )
                                    - (1 - (0.5 * (maxi + mini)))
                                )
                                * 100,
                                2,
                            )
                        )
                    ],
                    "Cote moyenne": [
                        sum(R_P_min_max["Cote" + str(self.equipe)]) / len(R_P_min_max)
                    ],
                },
                index=["Statistiques"],
            )

            Bien_Pred = []

            for Real, Pred in zip(
                R_P_min_max[self.target.columns[0]], R_P_min_max["Prediction"]
            ):

                if Real == Pred:
                    Bien_Pred.append(1)
                else:
                    Bien_Pred.append(0)

            R_P_min_max["Good"] = Bien_Pred

            good = R_P_min_max.loc[R_P_min_max["Good"] == 1]
            bad = R_P_min_max.loc[R_P_min_max["Good"] == 0]

            bins = np.linspace(0.5, 1, 50)

            x = good[str(self.equipe)]
            y = bad[str(self.equipe)]
            plt.hist(x, bins, label=["Bien"], alpha=0.5, color="darkblue")
            plt.hist(y, bins, label=["Mauvais"], alpha=0.5, color="darkred")
            plt.legend(loc="upper right")
            plt.title("Fréquence de bonnes prédictions selon la cote", color="darkblue")
            plt.show()

            display(Stats)

        sldr = widgets.FloatSlider(value=0.5, min=0.5, max=1, step=0.01)
        sldr2 = widgets.FloatSlider(value=1, min=0.5, max=1, step=0.01)
        widgets.interact(Visualisation, mini=sldr, maxi=sldr2)
