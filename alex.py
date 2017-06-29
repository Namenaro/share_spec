# -*- coding: utf-8 -*
import theano
import theano.tensor
import lasagne
floatX = theano.config.floatX

from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy, squared_error
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import matplotlib.pyplot as plt
import itertools


from sklearn.datasets.samples_generator import make_blobs, make_moons
from sklearn.preprocessing import scale
import numpy as np

rng = np.random.RandomState(0)

class Magia:
    def __init__(self):
        self.input_var = theano.tensor.matrix('input_var')
        self.target_var = theano.tensor.vector('target_var')
        self.model = self.symbolic_model()

    def symbolic_model(self):
        input_layer = InputLayer(shape=(None,1),
                                 name='input_layer',
                                 input_var=self.input_var)

        num_hidden_neurons = 10
        second_layer = DenseLayer(incoming=input_layer,
                                  num_units=num_hidden_neurons,
                                  nonlinearity=lasagne.nonlinearities.rectify,
                                  name='second_layer')

        num_classes = 1
        output_layer = DenseLayer(incoming=second_layer,
                                num_units=num_classes,
                                nonlinearity=theano.tensor.nnet.sigmoid,
                                name='output_layer')

        return output_layer

    def get_test_data(self, num_samples):
        X, Y = self.regression_data(num_samples=num_samples)
        return X, Y

    def classification_data(self, num_samples):
        X, Y = make_moons(noise=0.2, random_state=0, n_samples=num_samples)
        X = X.astype(floatX)
        Y = Y.astype(floatX)
        return X, Y

    def regression_data(self, num_samples):
        gen = DataGen()
        X, Y = gen.make_data(num_samples=num_samples)
        X = np.array(X).astype(floatX)
        Y = np.array(Y).astype(floatX)
        return X, Y

    def get_train_data(self, num_samples):
        return self.get_test_data(num_samples=num_samples)

    def define_train_fn(self):
        # символьная оптимизируемая функция
        predictions = get_output(self.model)
        loss = squared_error(predictions, self.target_var)  # возвращает 1d тензор
        loss = loss.mean()  # а вот теперь скаляр

        # какие параметры оптимизируем и как
        params = lasagne.layers.get_all_params(self.model, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
        train_fn = theano.function(inputs=[self.input_var, self.target_var],
                                   outputs=loss,
                                   updates=updates,
                                   allow_input_downcast=True)  # float64 ->float32
        return train_fn

    def define_test_fn(self):
        test_prediction = get_output(self.model, deterministic=True)
        test_loss = squared_error(test_prediction,self.target_var)
        test_loss = test_loss.mean()
        test_acc = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), self.target_var),
                          dtype=theano.config.floatX)
        val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])
        return val_fn

    def main_cycle(self):
        train_fn = self.define_train_fn()
        val_fn = self.define_test_fn()
        for i in range(300):
            data, targets = self.get_train_data(num_samples=5)
            data = np.matrix(data).T
            targets = np.array(targets)
            train_fn(data, targets)

            data, targets = self.get_test_data(num_samples=5)
            data = np.matrix(data).T
            targets = np.array(targets)
            err, acc = val_fn(data, targets)
            print "Test:" + ", err=" + str(err)

    def make_prediction(self, ):
        X = np.linspace(-5, 5, 10)
        X = np.array(X).astype(floatX)
        X = np.matrix(X).T
        x = theano.tensor.matrix('x', dtype=theano.config.floatX)
        prediction = lasagne.layers.get_output(self.model, x)
        f = theano.function([x], prediction)
        output = f(X)

        print "make_pred:"
        print(output)
        plt.figure()
        X = np.asarray(X).reshape(-1)
        output = output.flatten()

        plt.scatter(X, output, c='k', label='data', zorder=1)
        plt.show()



class DataGen:
    def __init__(self):
        pass

    def make_Y(self, X, noise):
        Y = []
        for x in X:
            Y.append(self.make_y(x, noise))
        return Y

    def make_y(self, x, noise):
        y = np.sin(x) + noise * rng.rand()
        return y

    def make_X(self, mu, sigma, n):
        X= []
        for i in range(n):
            x = np.random.normal(mu, sigma)
            X.append(x)
        return X

    def make_data(self, num_samples):
        n1 = int(num_samples/2)
        n2 = num_samples - n1
        X1 = self.make_X(mu=0.0, sigma=0.3, n=n1)
        X2 = self.make_X(mu=2.5, sigma=0.6, n=n2)
        Y1 = self.make_Y(X1, noise=0.6)
        Y2 = self.make_Y(X2, noise=0.00)
        X = X1 + X2
        Y = Y1 + Y2
        return X, Y

    def show_data(self, X, Y):
        plt.figure()
        print str(X)
        print str(Y)
        plt.scatter(X, Y, c='k', label='data', zorder=1)
        plt.show()

if __name__ == "__main__":
    magia = Magia()
    magia.main_cycle()
    magia.make_prediction()











