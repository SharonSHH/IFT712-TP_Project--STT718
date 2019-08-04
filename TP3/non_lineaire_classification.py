# -*- coding: utf-8 -*-
"""
Execution dans un terminal

Exemple:
   python non_lineaire_classification.py rbf 100 200 0 0

# DUHAMEL Thibault 18026048
# SHI Heng 18171434
"""

import numpy as np
import sys
from map_noyau import MAPnoyau
import gestion_donnees as gd


def analyse_erreur(err_train, err_test):
	"""
	Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
	apprentissage
	"""
	#10% safe margin
	if err_test>err_train+10:
		print("WARNING : Possible overfitting")
	if err_test>25 and err_train>25:
		print("WARNING : Possible underfitting")

def main():

	if len(sys.argv) < 6:
		usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
		\n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
		\n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
		\n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
		\n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
		print(usage)
		return

	type_noyau = sys.argv[1]
	nb_train = int(sys.argv[2])
	nb_test = int(sys.argv[3])
	lin_sep = int(sys.argv[4])
	vc = bool(int(sys.argv[5]))

	# On génère les données d'entraînement et de test
	generateur_donnees = gd.GestionDonnees(nb_train, nb_test, lin_sep)
	[x_train, t_train, x_test, t_test] = generateur_donnees.generer_donnees()

	# On entraine le modèle
	mp = MAPnoyau(noyau=type_noyau)

	if vc is False:
		mp.entrainement(x_train, t_train)
	else:
		mp.validation_croisee(x_train, t_train)

	err_x_train = [mp.erreur(t_train[i],mp.prediction(x_train[i])) for i in range(x_train.shape[0])]
	err_x_test = [mp.erreur(t_test[i],mp.prediction(x_test[i])) for i in range(x_test.shape[0])]
	err_train = 100*np.sum(err_x_train)/x_train.shape[0]
	err_test = 100*np.sum(err_x_test)/x_test.shape[0]

	print('Erreur train = ', err_train, '%')
	print('Erreur test = ', err_test, '%')
	analyse_erreur(err_train, err_test)

	# Affichage
	mp.affichage(x_test, t_test)

if __name__ == "__main__":
	main()
