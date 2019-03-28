import pickle
import numpy as np
import collections
from decisiontree import DecisionTree
import matplotlib.pyplot as plt

# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[data, id2titles, fields] = pickle.load(open("imdb_extrait.pkl", "rb"))
# la derniere colonne est le vote
datax = data[:, :32]
datay = np.array([1 if x[33] > 6.5 else -1 for x in data])


# Q 1.1
def entropie(vect):
	cnt = collections.Counter()
	for i in vect:
		cnt[i] += 1
	H = 0
	for c in cnt:
		H += (cnt[c] / len(vect)) * np.log2(cnt[c] / len(vect))
	return -H


# Q 1.2
def entropie_cond(list_vect):
	sizes = np.zeros(len(list_vect))
	entropies = np.zeros(len(list_vect))
	for i, vect in enumerate(list_vect):
		sizes[i] = len(vect)
		entropies[i] = entropie(vect)
	totalSize = np.sum(sizes)
	return np.sum((sizes / totalSize) * entropies)


# Quelques tests des fonctions
# print(entropie([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
# print(entropie([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
# print(entropie([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
# print(entropie_cond([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]))


# Mettre True pour l'entropie
if True:
	# Q 1.3
	entr = entropie(datay)
	entrc = np.zeros(29)
	print("Entropie : " + str(entr))
	for i in range(0, 28):
		entrc[i] = entropie_cond([datay[np.where(datax[:, i] == 1)[0]], datay[np.where(datax[:, i] != 1)[0]]])
		print(fields[i] + "\t" + " : Entropie conditionnelle:\t" + str(entrc[i]) + "\t | difference : " + str(
			entr - entrc[i]))
# Valeur de 0 : Le vecteur est homogène
# Valeur de 1 : Le vecteur possède autant de 1 que de 0
# Le meilleur attribut pour la première partition est celui qui apporte plus plus d'informations, donc celui dont la différence d'entropie est la plus grande


# Q 1.4
# PDF des graphes joints à ce fichier
# On sépare beaucoup d'éléments avec les premières séparations et plus on descent, plus le nombre d'exemples devient petit
# On filtre au fur et à mesure, c'est donc normal

# Mettre True pour Quelques expériences préliminaires
if True:
	dt = DecisionTree()
	dt.max_depth = 1  # on fixe la taille de l ’ arbre a 5
	dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud
	dt.fit(datax, datay)
	dt.predict(datax[:5, :])
	print(dt.score(datax, datay))
	# dessine l ’ arbre dans un fichier pdf si pydot est installe .
	dt.to_pdf("/tmp/test_tree.pdf", fields)


# sinon utiliser http :// www . webgraphviz . com /
# dt.to_dot(fields)
# ou dans la console
# print(dt.print_tree(fields))

# Q 1.5
# Profondeur 1 : 0.64
# Profondeur 3 : 0.72
# Profondeur 5 : 0.74
# Profondeur 8 : 0.79
# Plus la profondeur augmente, plus le score augmente. C'est normal car on distingue de plus en plus de cas.
# Avec une profondeur infinie on ne trouverait que des feuilles homogènes, donnant 100% de score

# Q 1.6
# Ce score n'est pas fiable car on test sur la base d'apprentissage.
# Plus on augmente la complexité de la séparation, plus on apprend par coeur sans pouvoir s'adapter.

# Q 1.7
def apprentissage(datax, datay, prop):
	ax = datax[:int(np.floor(prop * len(datax)))]  # donnee apprentissage
	ay = datay[:int(np.floor(prop * len(datax)))]

	tx = datax[int(np.floor(prop * len(datax))):]  # donnee test
	ty = datay[int(np.floor(prop * len(datax))):]

	ascore = np.zeros(9)
	tscore = np.zeros(9)

	for d in range(1, 28, 3):
		print("apprentissage : prop = " + str(prop) + " depth = " + str(d))
		dt = DecisionTree()
		dt.max_depth = d  # on fixe la taille de l ’ arbre a 5
		dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud
		dt.fit(ax, ay)
		ascore[int(np.floor(d / 3))] = 1 - dt.score(ax, ay)
		tscore[int(np.floor(d / 3))] = 1 - dt.score(tx, ty)
	plt.plot(range(1, 28, 3), ascore)
	plt.plot(range(1, 28, 3), tscore)
	plt.legend(["Apprentissage", "Test"])
	plt.title("Proportion : " + str(prop))
	plt.show()


# Mettez à True pour Sur et sous apprentissage
if True:
	# Partage 0.2, 0.8
	apprentissage(datax, datay, 0.2)
	# Partage 0.5, 0.5
	apprentissage(datax, datay, 0.5)
	# Partage 0.8, 0.2
	apprentissage(datax, datay, 0.8)

# Q 1.8
# L'erreur en apprentissage diminue rapidement, pour se retrouver proche de 0 à une profondeur élevée
# L'erreur de test diminue aussi mais réaugmente plus on apprend le modèle profondément
# Avec peu de données en apprentissage, l'erreur en apprentissage diminue plus rapidement, l'erreur en test reste relativement constante
# Avec beaucoup de données en apprentissage, l'erreur en apprentissage diminue plus lentement mais
# l'erreur de test atteint un minimum plus bas mais augmente plus rapidement

# Q 1.9
# Les résultats semblent cohérents, mais ils ne sont pas fiables car la sélection des données d'apprentissage peut être biaisé


# Mettre à True pour Validation croisée
if True:
	n = 4
	indices = np.array(range(len(datax)))
	np.random.shuffle(indices)
	minErr = np.inf
	model_depth = 1
	for d in range(1, 22, 3):
		print("Testing model depth :" + str(d))
		currError = 0
		for i in range(n):
			rind = indices[int(np.floor(i * (len(datax) / 5))):int(np.floor((i + 1) * (len(datax) / 5)))]
			ax = np.delete(datax, rind, axis=0)
			ay = np.delete(datay, rind, axis=0)
			tx = datax[rind]
			ty = datay[rind]

			dt = DecisionTree()
			dt.max_depth = d
			dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud
			dt.fit(ax, ay)
			currError = currError + (1 - dt.score(tx, ty)) / n
		print("Error : " + str(currError))
		if currError < minErr:
			model_depth = d
			minErr = currError
	print("\nOptimal depth : " + str(model_depth))
