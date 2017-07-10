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
import lasagne.layers


from sklearn.datasets.samples_generator import make_blobs, make_moons
from sklearn.preprocessing import scale
import numpy as np
import alex_data

rng = np.random.RandomState(0)

class Magia:
    def __init__(self, data_generator):
        self.params = {}
        self.params['weight_decay'] = 0.0001
        self.params['learning_rate'] = 0.02
        self.params['num_iterations'] = 3000
        self.params['batch_size'] = 5
        self.params['dropout'] = 0.3
        self.params['num_neurons'] = 6
        self.gen = data_generator
        self.input_var = theano.tensor.matrix('input_var')
        self.target_var = theano.tensor.vector('target_var')
        self.model = self.symbolic_droput_model()


    def symbolic_droput_model(self):
        input_layer = InputLayer(shape=(None, 1),
                                 name='input_layer',
                                 input_var=self.input_var)

        d2 = DenseLayer(incoming=input_layer,
                                  num_units=self.params['num_neurons'],
                                  nonlinearity=lasagne.nonlinearities.rectify,
                                  name='second_layer')

        dr2 = lasagne.layers.DropoutLayer(d2, p=self.params['dropout'])

        d3 = DenseLayer(incoming=dr2,
                        num_units=self.params['num_neurons'],
                        nonlinearity=lasagne.nonlinearities.rectify,
                        name='second_layer')

        dr3 = lasagne.layers.DropoutLayer(d3, p=self.params['dropout'])

        output_layer = DenseLayer(incoming=dr3,
                                  num_units=1,
                                  nonlinearity=theano.tensor.tanh,
                                  name='output_layer')

        return output_layer

    def get_data(self, num_samples):
        X, Y = self.gen.get_batch(num_samples)
        X = np.array(X).astype(floatX)
        Y = np.array(Y).astype(floatX)
        return X, Y

    def get_test_data(self, num_samples):
        X, Y = self.gen.get_test_data(5)
        X = np.array(X).astype(floatX)
        Y = np.array(Y).astype(floatX)
        return X, Y

    def define_train_fn(self):
        # символьная оптимизируемая функция
        predictions = get_output(self.model)
        loss = squared_error(predictions, self.target_var)  # возвращает 1d тензор
        loss = loss.mean()  # а вот теперь скаляр
        weights_L2 = lasagne.regularization.regularize_network_params(self.model, lasagne.regularization.l2)
        loss += self.params['weight_decay'] * weights_L2

        # какие параметры оптимизируем и как
        params = lasagne.layers.get_all_params(self.model, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.02, momentum=0.9)
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
        for i in range(self.params['num_iterations']):
            data, targets = self.get_data(num_samples=self.params['batch_size'])
            data = np.matrix(data).T
            targets = np.array(targets)
            train_fn(data, targets)

            data, targets = self.get_test_data(5)
            data = np.matrix(data).T
            targets = np.array(targets)
            err, acc = val_fn(data, targets)
            print "Test:" + ", err=" + str(err)

    def make_prediction(self):
        X = np.linspace(-5, 5, 100)
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
        plt.scatter(self.gen.X, self.gen.Y, color='red', label='real', zorder=1)


    def make_pred_in_one_point(self, X):
        # получает точку входного пространства
        # возвращает ответ сети в этой точке
        X = np.array(X).astype(floatX)
        X = np.matrix(X).T
        x = theano.tensor.matrix('x', dtype=theano.config.floatX)
        prediction = lasagne.layers.get_output(self.model, x, deterministic=False)
        f = theano.function([x], prediction)
        output = f(X)
        return output

    def make_distribution_in_point(self, x, n_samples):
        samples = []
        for _ in range(n_samples):
            ans = self.make_pred_in_one_point(x)
            samples.append(ans)
        plt.figure()
        a = np.empty(n_samples)
        a.fill(x)
        plt.scatter(a, samples, color='red', label='real', zorder=1)
        plt.show()
        return samples

    def make_full_pred(self, n_samples_per_point, from_x, to_x, n_points):
        plt.figure()
        x = from_x
        dx = (to_x - from_x) /float(n_points)
        print "dx=" + str (dx)
        for i in range(n_points):
            print "next point.."
            samples = []
            for _ in range(n_samples_per_point):
                ans = self.make_pred_in_one_point(x)
                samples.append(ans)
            a = np.empty(n_samples_per_point)
            a.fill(x)
            plt.scatter(a, samples, color='red', label='real', zorder=1)
            x += dx
        plt.show()

def experiment( gen):
    for i in range(6,16,2):
        magia = Magia(gen)
        magia.params['num_neurons'] = i
        magia.params['num_iterations'] = i*3000
        magia.main_cycle()
        magia.make_prediction()
        del magia
        plt.savefig("neurons_"+str(i)+".png")

if __name__ == "__main__":
    generator = alex_data.AlexData()
    #magia = Magia(generator)
    #magia.main_cycle()
    #magia.make_prediction()
    #dis_x = magia.make_distribution_in_point(x=2., n_samples=30)
    #magia.make_full_pred(n_samples_per_point=4,
    #                     from_x=-5,
    #                     to_x=5,
    #                     n_points=15)
    experiment(generator)










