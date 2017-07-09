# -*- coding: utf-8 -*

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(0)

# Класс отвечающий за датасет для регресии из 1д в 1д.
# Класс инициализирует датасет
# Выдает по запросу случайное кол-во семплоов
# отрисовывает датасет
class AlexData:
    def __init__(self):
        self.X = []
        self.Y = []
        self.make_XY(20)

    def _make_y(self, x, noise):
        y = np.sin(x) + noise * rng.rand()
        return y

    def _make_Y(self, X, noise):
        Y = []
        for x in X:
            Y.append(self._make_y(x, noise))
        return Y

    def _make_X(self, mu, sigma, n):
        X = []
        for _ in range(n):
            x = np.random.normal(mu, sigma)
            X.append(x)
        return X

    def make_XY(self, num_samples):
        n1 = int(num_samples/2)
        n2 = num_samples - n1
        X1 = self._make_X(mu=0.0, sigma=0.3, n=n1)
        X2 = self._make_X(mu=2.5, sigma=0.6, n=n2)
        Y1 = self._make_Y(X1, noise=0.6)
        Y2 = self._make_Y(X2, noise=0.00)
        self.X = X1 + X2
        self.Y = Y1 + Y2

    def show_XY(self, X=None, Y=None):
        if X is None:
            X = self.X
            Y = self.Y
        plt.figure()
        print str(X)
        print str(Y)
        plt.scatter(X, Y, c='k', label='data', zorder=1)
        plt.show()

    def get_batch(self, size):
        assert size <= len(self.X)
        assert len(self.X) == len(self.Y)
        indexes = np.random.choice(len(self.X), size)
        batchX = []
        batchY = []
        for index in indexes:
            batchX.append(self.X[index])
            batchY.append(self.Y[index])
        return batchX, batchY

if __name__ == "__main__":
     data = AlexData()
     data.show_XY()
     X,Y = data.get_batch(10)
     data.show_XY(X=X, Y=Y)
     X, Y = data.get_batch(10)
     data.show_XY(X=X, Y=Y)
     X, Y = data.get_batch(10)
     data.show_XY(X=X, Y=Y)




