# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        if np.isscalar(x):
            phi_x = np.vander([x], self.M, increasing=True)
        else:
            phi_x = np.vander(x, self.M, increasing=True)
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI
        print("Recherche de M...")
        splits_len = X.shape[0]/10

        best_error = float("inf")
        best_M = 1
        for k in range(10):
            X_validation = X[int(k*splits_len) : int((k+1)*splits_len)]
            X_train = np.concatenate((X[:int(k*splits_len)],X[int((k+1)*splits_len):]), axis=0)
            t_validation = t[int(k*splits_len) : int((k+1)*splits_len)]
            t_train = np.concatenate((t[:int(k*splits_len)],t[int((k+1)*splits_len):]), axis=0)
            self.M = k+1
            self.entrainement(X_train, t_train, False)
            error_validation = Regression.erreur(self.prediction(X_validation), t_validation)
            print("M = " + str(k+1) + ", Validation Loss = " + str(error_validation))
            if (error_validation < best_error):
                best_error = error_validation
                best_M = k+1
        print("Best M : " + str(best_M))
        self.M = best_M


    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        if (using_sklearn):
            ridge = linear_model.Ridge(alpha=self.lamb, fit_intercept=False).fit(phi_x,t)
            weights = ridge.coef_
        else:
            weights = np.linalg.solve(np.dot(np.transpose(phi_x),phi_x)+self.lamb*np.identity(self.M), np.dot(np.transpose(phi_x),t))

        self.w = weights

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        pred = np.dot(self.fonction_base_polynomiale(x),self.w)
        if pred.shape == (1,):
            return pred[0]
        else:
            return pred

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        if np.isscalar(t):
            return (t-prediction)**2
        else:
            return np.mean((t-prediction)**2)
