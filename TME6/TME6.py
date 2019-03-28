import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
from arftools import *
import random


def plot_frontiere_proba(data, f, step=20):
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255)


def score(svm, datax, datay):
    return np.mean(svm.predict(datax) == datay)


print("Lineairement séparable")
# Lineairement separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=0, epsilon=1)
testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=1)

# lineaire avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='linear')
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))

# lineaire avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='linear', C=99)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))

print("Non lineaire")
# Non-lineairement separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=1, epsilon=0.1)
testx, testy = gen_arti(nbex=1000, data_type=1, epsilon=0.1)

# lineaire avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='linear', max_iter=2000)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))

# lineaire avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='linear', C=99, max_iter=2000)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))

# polynomial avec C par défaut
svm = sklearn.svm.SVC(probability=True, kernel='poly', degree=2, gamma='scale', max_iter=2000)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Polynomial : ", score(svm, testx, testy))

print("Echiquier")
# Echiquier separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=2, epsilon=0.001)
testx, testy = gen_arti(nbex=1000, data_type=2, epsilon=0.001)

# Poly avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='poly', degree=2, gamma='scale', max_iter=2000)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Polynomial : ", score(svm, testx, testy))

# Noyau gaussien
svm = sklearn.svm.SVC(probability=True, kernel='rbf', gamma=0.08, C=10000)
svm.fit(datax, datay)

plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Gaussien : ", score(svm, testx, testy))

# Grid search
if True : # Mettre à True pour effectuer un grid search
    gammas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    Cs = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    indices = range(len(datay))
    mscore = 0
    bestg = gammas[0]
    bestc = Cs[0]
    for g in gammas:
        for c in Cs:
            print("Testing g=", g, " c=", c)
            svm = sklearn.svm.SVC(probability=True, kernel='rbf', gamma=g, C=c)
            ind = random.sample(indices, int(len(indices)/len(Cs)))
            svm.fit(np.delete(datax, ind,axis=0), np.delete(datay, ind, axis=0))
            sc = svm.score(datax[ind, :], datay[ind])
            if sc > mscore:
                mscore = sc
                bestg = g
                bestc = c

    print("Meilleur : g=", bestg, " c=", bestc)
    svm = sklearn.svm.SVC(probability=True, kernel='rbf', gamma=bestg, C=bestc)
    svm.fit(datax, datay)
    plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
    plot_data(datax, datay)
    plt.show()
    print("Gaussien VC : ", score(svm, testx, testy))


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


datax, datay = load_usps("USPS_train.txt")
tx, ty = load_usps("USPS_test.txt")

#Multi classes
svms = np.ndarray((10, 10)).astype(sklearn.svm.SVC)
for i in range(9):
    for j in range(i+1, 10):
        svm = sklearn.svm.SVC(kernel='linear')
        svms[i, j] = svm
        indi = datay == i
        indj = datay == j
        ind = np.bitwise_or(indi, indj)
        svm.fit(datax[ind, :], 2 * (datay[ind] == i) - 1)

predicts = np.zeros((len(ty), 10))
for i in range(9):
    for j in range(i+1, 10):
        pred = svms[i, j].predict(tx)
        predicts[:, i] += pred >= 0
        predicts[:, j] += pred < 0

print("Multi classes, One vs one, score : ", np.mean(ty == np.argmax(predicts, axis=1)))
svms = []
for i in range(10):
    svm = sklearn.svm.SVC(kernel='linear')
    svms.append(svm)
    svm.fit(datax, 2*(datay==i)-1)

predicts = np.zeros((len(ty), 10))
for i in range(10):
    predicts[:, i] = svms[i].predict(tx)
print("Multi classes, One vs all, score : ", np.mean(ty == np.argmax(predicts, axis=1)))

#One vs one est plus rapide et plus précis, mais est un peuplus complexe à mettre en place



