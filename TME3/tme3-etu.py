import sys

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def make_grid(xmin=-5, xmax=5, ymin=-5, ymax=5, step=20, data=None):
	""" Cree une grille sous forme de matrice 2d de la liste des points
	:return: une matrice 2d contenant les points de la grille, la liste x, la liste y
	"""
	if data is not None:
		xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), \
		                         np.max(data[:, 1]), np.min(data[:, 1])
	x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
	                   np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
	grid = np.c_[x.ravel(), y.ravel()]
	return grid, x, y


def load_usps(filename):
	with open(filename, "r") as f:
		f.readline()
		data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
	tmp = np.array(data)
	return tmp[:, 1:], tmp[:, 0].astype(int)


# Partie 1
def optimize(fonc, dfonc, xinit, eps, max_iter):
	x = xinit
	x_histo = []
	f_histo = []
	grad_histo = []
	for i in range(max_iter):
		x_histo.append(x)
		f_histo.append(fonc(x))
		g = dfonc(x)
		grad_histo.append(g)
		x = x - eps * g
	return np.array(x_histo), np.array(f_histo), np.array(grad_histo)


# Partie 2
def xcosx(x):
	return x * np.cos(x)


def dxcosx(x):
	return np.cos(x) - x * np.sin(x)


def func2(x):
	return -np.log(x) + x * x


def dfunc2(x):
	return 2 * x - 1 / x


def rosenbrock(x):
	if len(np.shape(x)) == 2:
		x1 = x[:, 0]
		x2 = x[:, 1]
	else:
		x1 = x[0]
		x2 = x[1]
	x2x1 = x2 - x1 * x1
	onex1 = np.ones(np.shape(x1)) - x1
	return 100 * x2x1 * x2x1 + onex1 * onex1


def drosenbrock(x):
	if len(np.shape(x)) == 2:
		x1 = x[:, 0]
		x2 = x[:, 1]
	else:
		x1 = x[0]
		x2 = x[1]

	g = np.zeros(np.shape(x))
	if len(np.shape(x)) == 2:
		one = np.ones(np.shape(x1))
		g[:, 0] = 2 * (200 * x1 * x1 * x1 - 200 * x1 * x2 + x1 - one)
		g[:, 1] = 200 * (x2 - x1 * x1)
	else:
		g[0] = 2 * (200 * x1 * x1 * x1 - 200 * x1 * x2 + x1 - 1)
		g[1] = 200 * (x2 - x1 * x1)
	return g


def plotStuff(f, df, plotlimmin, plotlimmax, eps, maxite, plotTrajectory3D):
	if plotTrajectory3D:
		init = np.array(
			[np.random.uniform(plotlimmin[0], plotlimmax[0]), np.random.uniform(plotlimmin[1], plotlimmax[1])])
	else:
		init = np.random.uniform(plotlimmin, plotlimmax)
	xh, fh, gh = optimize(f, df, init, eps, maxite)
	if not plotTrajectory3D:
		plt.figure()
		plt.plot(range(maxite), fh)
		plt.plot(range(maxite), gh)
		plt.legend(["f", "gradient"])
		plt.show()

	if plotTrajectory3D:
		grid, xx, yy = make_grid(-2, 2, -1, 3, 20)
		plt.figure()
		plt.contourf(xx, yy, f(grid).reshape(xx.shape))
		plt.plot(xh[:, 0], xh[:, 1])
		plt.show()
	else:
		plt.figure()
		x = np.linspace(plotlimmin, plotlimmax, 200)
		plt.plot(x, f(x))
		plt.plot(xh, fh)
		plt.legend(["f", "f_histo"])
		plt.show()

	plt.figure()
	xstar = xh[np.argmin(fh)]
	plt.plot(range(maxite), [np.log(np.linalg.norm(xh[i] - xstar)) for i in range(maxite)])
	plt.show()


plotStuff(xcosx, dxcosx, -3, 3, 0.05, 100, False)
plotStuff(func2, dfunc2, 0.1, 2, 0.05, 100, False)

mafonction = rosenbrock
grid, xx, yy = make_grid(-2, 2, -1, 3, 20)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, rosenbrock(grid).reshape(xx.shape), rstride=1, cstride=1, \
                       cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.show()
plotStuff(rosenbrock, drosenbrock, [-2, -1], [2, 3], 0.001, 100, True)

# Regression logistique
datax, datay = load_usps("USPS_train.txt")
tx, ty = load_usps("USPS_test.txt")


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class USPS_reg:
	def __init__(self, nbIte, eps):
		self.weights = np.zeros(0)
		self.eps = eps
		self.nbIte = nbIte

	def fit(self, data, y):
		numClass = len(set(y))
		data = np.hstack((data, np.ones((len(data), 1))))
		N = len(data)
		self.weights = np.zeros((numClass, len(data[0] + 1)))
		yi = np.zeros((numClass,N))
		for i in range(numClass):
			yi[i,:] = (y==i)*1
		for it in range(self.nbIte):
			pred = sigmoid(np.dot(self.weights, data.T))
			diff = pred-yi
			grad = np.dot(diff, data)
			self.weights = self.weights - self.eps*grad/N

	def predict(self, data):
		data = np.hstack((data, np.ones((len(data), 1))))
		return np.argmax(np.dot(self.weights, data.T), axis=0)

	def score(self, datax, datay):
		return np.mean((self.predict(datax) == datay)*1)

class bayes:
	def __init__(self):
		self.weights = np.zeros(0)

	def fit(self, data, y):
		numClass = len(set(y))
		N = len(data[0])
		self.weights = np.zeros((numClass, N))
		for i in range(numClass):
			datai = data[np.where(y == i)]
			s = np.sum(datai, axis=0)
			self.weights[i, :] = s/len(datai)

	def predict(self, data):
		return np.argmax(np.dot(self.weights, data.T), axis=0)

	def score(self, datax, datay):
		return np.mean((self.predict(datax) == datay)*1)



reg = USPS_reg(100, 0.1)
np.set_printoptions(threshold=sys.maxsize)
print("Apprentissage regression")
reg.fit(datax, datay)
bay = bayes()
print("Apprentissage bayes")
bay.fit(datax, datay)
print("Regression logistique :")
print("Score en apprentissage : " + str(reg.score(datax, datay)))
print("Score en test : " + str(reg.score(tx, ty)))
print("Bayes naif :")
print("Score en apprentissage : " + str(bay.score(datax, datay)))
print("Score en test : " + str(bay.score(tx, ty)))
