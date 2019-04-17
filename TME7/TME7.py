import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        #calcul le cout
        pass
    def backward(self, y ,yhat):
        #calcul le gradient du cout
        pass

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None
    def zero_grad(self):
        #Annule le gradient
        pass
    def forward(self,X):
        #calcule la passe forward
        pass
    def update_parameters(self, gradient_step):
        #calcule la maj des poids
        pass
    def backward_update_gradient(self, input, delta):
        #met à jour la valeur du gradient
        pass
    def backward_delta(self, input, delta):
        #calcule la dérivée de l'erreur
        pass

class Lineaire(Module):
    def __init__(self, numberOfInput, numberOfOutput):
        super().__init__()
        self._parameters = 2*np.random.rand(numberOfOutput, numberOfInput+1)-1
        self._gradient = np.zeros((numberOfOutput, numberOfInput+1))

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)

    def forward(self, X):
        return np.matmul(self._parameters, np.hstack((1, X)))

    def update_parameters(self, gradient_step):
        super().update_parameters(gradient_step)

    def backward_update_gradient(self, input, delta):
        super().backward_update_gradient(input, delta)

    def backward_delta(self, input, delta):
        super().backward_delta(input, delta)

class Mtanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        super().backward_delta(input, delta)

class Msign(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.sign(X)

    def backward_delta(self, input, delta):
        return delta


class MSE(Loss):
    def forward(self, y, yhat):
        diff = yhat-y
        return np.mean(np.multiply(diff, diff))

    def backward(self, y, yhat):
        return 2*(yhat-y)/len(y)


#Test de forward avec le XOR

datax = np.asarray([[1,1], [1, -1], [-1, 1], [-1, -1]])
datay = np.asarray([-1, 1, 1, -1])

m1 = Lineaire(2, 2)
mt = Msign()
m2 = Lineaire(2, 1)
m1._parameters = np.asarray([[-1, -1, 1], [-1, 1, -1]])
m2._parameters = np.asarray([1, 1, 1])

for x in datax:
    a1 = m1.forward(x)
    z1 = mt.forward(a1)
    a2 = m2.forward(z1)
    z2 = mt.forward(a2)
    print(str(x) + " : " + str(z2))


#Test d'apprentissage du XOR

datax = np.asarray([[1,1], [1, -1], [-1, 1], [-1, -1]])
datay = np.asarray([[-1], [1], [1], [-1]])

m1 = Lineaire(2, 2)
mt = Mtanh()
l = MSE()
m2 = Lineaire(2, 1)

for i in range(500):
    for x in datax:
        a1 = m1.forward(x)
        z1 = mt.forward(a1)
        a2 = m2.forward(z1)
        z2 = mt.forward(a2)
