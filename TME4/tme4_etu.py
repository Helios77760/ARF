from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean((np.dot(datax, w.T) - datay) ** 2)


def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    return -np.mean(np.dot(datax.T, np.dot(datax, w.T) - datay))


def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
    return np.mean(np.max([np.zeros((np.shape(datay))), - datay * np.dot(datax, w.T)], axis=1))


def hinge_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur hinge """
    #    grad = np.zeros(datax.shape)
    #    scal = datay * np.dot(datax, w.T)
    #    ind = np.where(scal <= 0)
    #    grad[ind,:] = -datay[ind] * datax[ind,:]
    #    return np.mean(grad, axis=0)

    return np.mean(np.dot(datax.T, datay*(datay * np.dot(datax, w.T) < 0)).T, axis=0)


class Lineaire(object):
    def __init__(self, loss=hinge, loss_g=hinge_g, max_iter=1000, eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g
        self.whisto=0
        self.w = None

    def fit(self, datax, datay, testx=None, testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1, 1)
        N = len(datay)
        datax = datax.reshape(N, -1)
        D = datax.shape[1]
        self.whisto = np.ndarray((self.max_iter + 1, D))
        if self.w is None:
            self.w = np.random.random((1, D))
        self.whisto[0, :] = self.w
        for i in range(self.max_iter):
            self.w = self.w + self.loss_g(datax, datay, self.w) * self.eps
            self.whisto[i+1, :] = self.w

    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)
        return np.dot(datax, self.w.T)

    def score(self, datax, datay):
        return np.mean((self.predict(datax).T * datay > 0) * 1.0)


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")


def plot_error(datax, datay, f, step=10):
    grid, x1list, x2list = make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4)
    plt.contourf(x1list, x2list, np.array([f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    plt.show()


def projectionGaussienne(data, grid, k, sigma):
    n = np.shape(data)[0]
    D = np.shape(data)[1]
    gsize = np.shape(grid)[0]
    proj = np.zeros((n, gsize))
    #gridx = np.reshape((grid[:, 0]), (1, gsize))
    #gridy = np.reshape((grid[:, 1]), (1, gsize))
    #gridx = np.repeat(gridx, n, axis=0)
    #gridy = np.repeat(gridy, n, axis=0)
    for i, d in enumerate(data):
        dr = np.repeat([d], gsize, axis=0)
        norm = np.linalg.norm(dr-grid, axis=1)
        proj[i, :] = k*np.exp(-(norm*norm)/sigma)
    return proj




if __name__ == "__main__":
    """ Tracer des isocourbes de l'erreur """
    # plt.ion()
    trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=1)

    plt.figure()
    plot_error(trainx, trainy, mse)
    plt.figure()
    plot_error(trainx, trainy, hinge)
    trainy.reshape((1, -1))
    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.0001)
    perceptron.fit(trainx, trainy)
    print("Erreur : train %f, test %f" % (perceptron.score(trainx, trainy), perceptron.score(testx, testy)))
    plt.figure()
    plot_frontiere(trainx, perceptron.predict, 200)
    plot_data(trainx, trainy)
    plt.show()
    plt.figure()
    plt.plot(perceptron.whisto[:, 0], perceptron.whisto[:, 1])
    plt.show()
    plt.figure()
    linx = np.array([min(trainx[:,0]), max(trainx[:, 0])])
    for h in perceptron.whisto:
        plt.plot(linx, -h[0]*linx/h[1], alpha=0.05, c="k")
    plt.ylim(min(trainx[:,1]), max(trainx[:, 1]))
    plt.show()
    datax, datay = load_usps("USPS_train.txt")
    #datax = np.hstack((datax, np.ones((np.shape(datax)[0], 1))))
    tx, ty = load_usps("USPS_test.txt")
    #tx = np.hstack((tx, np.ones((np.shape(tx)[0], 1))))
    perceptron = Lineaire(hinge, hinge_g, 1000, 0.0001)
    ind6_9 = (datay == 6) + (datay == 9)
    ind1_8 = (datay == 1) + (datay == 8)
    datay6_9 = datay[ind6_9]
    datay1_8 = datay[ind1_8]
    perceptron.fit(datax[ind6_9, :], np.ones(len(datay6_9))-2*(datay6_9 == 9))
    print("6 vs 9")
    print(perceptron.w)
    #perceptron.fit(datax[ind1_8, :], np.ones(len(datay1_8))-2*(datay1_8 == 8))
    print("1 vs 8")
    print(perceptron.w)
    #perceptron.fit(datax, np.ones(len(datay)) - 2 * (datay != 6))
    print("6 vs all")
    print(perceptron.w)

    step = 100
    iters = range(1, 1001, step)
    sc_train = np.zeros(len(iters))
    sc_test = np.zeros(len(iters))
    train_datay = np.ones(len(datay)) - 2 * (datay != 6)
    test_datay = np.ones(len(ty)) - 2*(ty != 6)
    ioff = 0
    perceptron = Lineaire(hinge, hinge_g, step, 0.00001)
    for iteration in iters:
        print(iteration)
        perceptron.fit(datax, train_datay)
        sc_train[ioff] = perceptron.score(datax, train_datay)
        sc_test[ioff] = perceptron.score(tx, test_datay)
        ioff+=1
    plt.figure()
    plt.plot(iters, sc_train)
    plt.plot(iters, sc_test)
    plt.legend(["Train", "Test"])
    plt.show()
    # On n'observe pas de sur-apprentissage


    train4x, train4y = gen_arti(nbex=1000, data_type=1, epsilon=0.05)
    plt.figure()
    perceptron = Lineaire(hinge, hinge_g, 500, 0.0001)
    perceptron.fit(train4x, train4y)
    plot_frontiere(train4x, perceptron.predict, 200)
    plot_data(train4x, train4y)
    plt.show()
    print("type 1 sans projection : ")
    print(perceptron.score(train4x, train4y))

    # Pas séparable linéairement => projection
    train4x = np.hstack((train4x, np.reshape((train4x[:, 0]*train4x[:, 1]), (np.shape(train4x)[0], 1)))) #Ajout de x1*x2
    perceptron = Lineaire(hinge, hinge_g, 500, 0.0001)
    perceptron.fit(train4x, train4y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pos = train4y >= 0
    neg = train4y < 0
    ax.scatter(train4x[pos, 0], train4x[pos, 1], train4x[pos, 2], c="g", marker="x")
    ax.scatter(train4x[neg, 0], train4x[neg, 1], train4x[neg, 2], c="r", marker="o")
    grid, xx,yy = make_grid(xmax=np.max(train4x[:,0]),  xmin=np.min(train4x[:,0]), ymax=np.max(train4x[:,1]), ymin=np.min(train4x[:,1]), step=10)
    weights = perceptron.w[0]
    z = (-weights[0]*xx - weights[1]*yy)*1.0/weights[2]
    surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.3)
    plt.show()
    print("type 1 avec projection : ")
    print(perceptron.score(train4x, train4y))

    # Projection gaussienne pour l'échiquier
    trainCheckx, trainChecky = gen_arti(nbex=1000, data_type=2, epsilon=0.005)
    testCheckx, testChecky = gen_arti(nbex=1000, data_type=2, epsilon=0.005)
    grid, xx, yy = make_grid(xmax=np.max(trainCheckx[:, 0]), xmin=np.min(trainCheckx[:, 0]),
                             ymax=np.max(trainCheckx[:, 1]), ymin=np.min(trainCheckx[:, 1]), step=20)
    projx = projectionGaussienne(trainCheckx, grid, 1, 0.1)
    projtestx = projectionGaussienne(testCheckx, grid, 1, 0.1)
    perceptron = Lineaire(hinge, hinge_g, 1000, 0.001)
    perceptron.fit(projx, trainChecky)
    print("type echiquier avec projection gaussienne sur grille")
    print(perceptron.score(projx, trainChecky))
    print(perceptron.score(projtestx, testChecky))
    gridtest, xx, yy = make_grid(xmax=np.max(trainCheckx[:, 0]), xmin=np.min(trainCheckx[:, 0]),
                             ymax=np.max(trainCheckx[:, 1]), ymin=np.min(trainCheckx[:, 1]), step=20)
    projtest = projectionGaussienne(gridtest, grid, 1, 0.1)
    res = (perceptron.predict(projtest) > 0).reshape((20, 20))
    plt.figure()
    plt.imshow(res, extent=[np.max(trainCheckx[:, 0]), np.min(trainCheckx[:, 0]), np.min(trainCheckx[:, 1]), np.max(trainCheckx[:, 1])], interpolation='none', \
               alpha=0.3)
    plot_data(trainCheckx, trainChecky)
    plt.show()
