import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

plt.close("all")
plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  ## coord_x min et max
ymin, ymax = 48.806, 48.916  ## coord_y min et max


def show_map():
	plt.imshow(parismap, extent=[xmin, xmax, ymin, ymax], aspect=1.5)


## extent pour controler l'echelle du plan


poidata = pickle.load(open("data/poi-paris.pkl", "rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI", ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]), 2))
for i, (k, v) in enumerate(poidata[typepoi].items()):
	geo_mat[i, :] = v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.8, s=3)


class Classifier(object):
	""" Classe generique d'un classifieur
		Dispose de 3 méthodes :
			fit pour apprendre
			predict pour predire
			score pour evaluer la precision
	"""

	def fit(self, data, y):
		raise NotImplementedError("fit non  implemente")

	def predict(self, data):
		raise NotImplementedError("predict non implemente")

	def score(self, data, y):
		return (self.predict(data) == y).mean()


class histoModel(Classifier):
	def __init__(self, steps):
		self.steps = steps
		self.hist = np.zeros((steps, steps))
		self.xstep = (xmax - xmin) / (self.steps - 1)
		self.ystep = (ymax - ymin) / (self.steps - 1)

	def fit(self, data, y):
		for d in data:
			x = int((d[1] - xmin) / self.xstep)
			y = int((d[0] - ymin) / self.ystep)
			self.hist[x, y] += 1
		self.hist = self.hist / len(data)

	def predict(self, data):
		result = np.zeros(len(data))
		for i, d in enumerate(data):
			x = int((d[0] - xmin) / self.xstep)
			y = int((d[1] - ymin) / self.ystep)
			result[i] = self.hist[x, y]
		return result


def parzen(x, y, h):
	return np.maximum(np.abs(x), np.abs(y)) / h < 0.5


class parzenModel(Classifier):
	def __init__(self, h):
		self.h = h
		self.data = np.zeros(0)
		self.v = 0

	def fit(self, data, y):
		self.data = data
		self.v = self.h ** len(self.data[0])

	def predict(self, data):
		result = np.zeros(len(data))
		for i, d in enumerate(data):
			diffx = self.data[:, 1] - d[0]
			diffy = self.data[:, 0] - d[1]
			ind = parzen(diffx, diffy, self.h)
			result[i] = np.sum(ind / self.v) / len(self.data)
		return result


def gauss(x, y, h):
	return np.exp(-0.5*(((x/h)**2)+(y/h)**2))/np.sqrt(2*np.pi)


class gaussModel(Classifier):
	def __init__(self, h):
		self.h = h
		self.data = np.zeros(0)

	def fit(self, data, y):
		self.data = data

	def predict(self, data):
		result = np.zeros(len(data))
		for i, d in enumerate(data):
			diffx = self.data[:, 1] - d[0]
			diffy = self.data[:, 0] - d[1]
			ind = gauss(diffx, diffy, self.h)
			result[i] = np.sum(ind) / len(self.data)
		return result


###################################################



# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx, yy = np.meshgrid(np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps))
grid = np.c_[xx.ravel(), yy.ravel()]

"""

#Modèle à histogramme
hModel = histoModel(40)
hModel.fit(geo_mat, [])
res = hModel.predict(grid).reshape(steps, steps)
# res = np.random.random((steps, steps))
plt.figure()
show_map()
plt.imshow(res, extent=[xmin, xmax, ymin, ymax], interpolation='none', \
           alpha=0.3, origin="lower", aspect=1.5)
plt.colorbar()
plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.3, s=3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


#Modèle des fenetres de Parzen
pModel = parzenModel(0.03)
pModel.fit(geo_mat, [])
res = pModel.predict(grid).reshape(steps, steps)
# res = np.random.random((steps, steps))
plt.figure()
show_map()
plt.imshow(res, extent=[xmin, xmax, ymin, ymax], interpolation='none', \
           alpha=0.3, origin="lower", aspect=1.5)
plt.colorbar()
plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.3, s=3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


#Modèle de fenetre gaussienne
gModel = gaussModel(0.005)
gModel.fit(geo_mat, [])
res = gModel.predict(grid).reshape(steps, steps)
# res = np.random.random((steps, steps))
plt.figure()
show_map()
plt.imshow(res, extent=[xmin, xmax, ymin, ymax], interpolation='none', \
           alpha=0.3, origin="lower", aspect=1.5)
plt.colorbar()
plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.3, s=3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# Une discretisation faible par la technique de l'histogramme ne donne pas de densité précise. La densité devient la même dans toute une case
# Une discrétisation trop forte crée des cases où la densité est très faible voire nulle alors que celle adjacente est très élevée,
# ce qui n'est pas très représentatif de la réalité

# Les paramètres des méthodes à noyaux permettent de définir la taille du noyau (pour parzen) ou encore l'écart type de la gaussienne
# Cela a pour conséquence de modifier la taille et l'importance de l'entourage de chaque point

# On peut choisir de manière automatique les paramètres par validation croisée

# On pourrait estimer la qualité du modèle en estimant la vraissemblance des points de tests par rapport à la densité en leur point

## Choix d'un poi
typepoi = "atm"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]), 2))
for i, (k, v) in enumerate(poidata[typepoi].items()):
	geo_mat[i, :] = v[0]

gModel = gaussModel(0.005)
gModel.fit(geo_mat, [])
res = gModel.predict(grid).reshape(steps, steps)
# res = np.random.random((steps, steps))
plt.figure()
show_map()
plt.imshow(res, extent=[xmin, xmax, ymin, ymax], interpolation='none', \
           alpha=0.3, origin="lower", aspect=1.5)
plt.colorbar()
plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.3, s=3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

"""

#############################################
# Predire la note en fonction de l'emplacement

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
gps_notes = np.zeros((len(poidata[typepoi]),3))
for i, (k, v) in enumerate(poidata[typepoi].items()):
	gps_notes[i, 0] = v[0][1] # coordonnée x
	gps_notes[i, 1] = v[0][0] # coordonnée y
	gps_notes[i, 2] = v[1]    # note

# Certains POI n'ont pas de note (noté comme ayant une note négative), on l'estimera en fonction des autres POI notés
a_estimer = gps_notes[np.where(gps_notes[:,2] < 0)]
gps_notes = gps_notes[np.where(gps_notes[:,2] >= 0)]

# Nadaraya-Watson

class NadarayaRegression(Classifier):
	def __init__(self, kernelFunction, h):
		self.kfunc = kernelFunction
		self.h = h
		self.data = np.zeros(0)
	def fit(self, data, y):
		self.data = data

	def predict(self, data):
		result = np.zeros(len(data))
		for i,d in enumerate(data):
			x = self.data[:, 0] - d[0]
			y = self.data[:, 1] - d[1]
			weight = self.kfunc(x, y, self.h)
			result[i] = np.sum(np.multiply(self.data[:, 2], weight))/np.sum(weight)
		return result


nadaraya = NadarayaRegression(gauss, 0.005)
nadaraya.fit(gps_notes, 0)
res = nadaraya.predict(a_estimer)
print(res)

# Knn

class KNN(Classifier):
	def __init__(self, k):
		self.k = int(k)
		self.data = np.zeros(0)
	def fit(self, data, y):
		self.data = data

	def predict(self, data):
		result = np.zeros(len(data))
		for i,d in enumerate(data):
			x = self.data[:, 0] - d[0]
			y = self.data[:, 1] - d[1]
			dist = np.sqrt(np.multiply(x,x)+np.multiply(y,y))
			nearestNeighborsIndices = np.argpartition(dist, self.k)[:self.k]
			result[i] = np.mean(self.data[nearestNeighborsIndices, 2])
		return result


knn = KNN(4)
knn.fit(gps_notes, 0)
res = knn.predict(a_estimer)
print(res)

