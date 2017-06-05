# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from model_visualiser import Visualizer

class EpisodicMemory:
    def __init__(self, enought_episodes_num):
        self.MAX_EPISODES = enought_episodes_num
        self.X = []
        self.Y = []

    def add_new_episode(self, X, Y):
        self.X.append(X[0])
        self.Y.append(Y[0])

    def contains_enough_memories(self):
        if len(self.X) >= self.MAX_EPISODES:
            return True
        return False

    def clean(self):
        self.X = []
        self.Y = []


class ModuleParams:
    """
    Параметры, для которых строятся апостериорные распределения
    """
    def __init__(self, n_input, n_hidden=5):
        self.W01_means = np.random.randn(n_input, n_hidden)
        self.W01_sds = np.random.random_sample((n_input, n_hidden))

        self.W12_means = np.random.randn(n_hidden)
        self.W12_sds = np.random.random_sample((n_hidden))

        self.b01_means = np.random.randn(n_hidden)
        self.b01_sds = np.random.random_sample(( n_hidden))


    def reset(self, v_params):
        self.W01_means = v_params.means['w01']
        self.W01_sds = v_params.stds['w01']
        self.W12_means = v_params.means['w12']
        self.W12_sds = v_params.stds['w12']
        self.b01_means = v_params.means['b01']
        self.b01_sds = v_params.stds['b01']

    def print_params_to_file(self, filename):
        filename = filename + '.txt'
        with file(filename, 'w') as outfile:
            outfile.write('# W01_sds shape: {0}\n'.format(self.W01_sds.shape))
            np.savetxt(outfile, self.W01_sds, fmt='%-7.2f')
            outfile.write('# W12_sds shape: {0}\n'.format(self.W12_sds.shape))
            np.savetxt(outfile, self.W12_sds, fmt='%-7.2f')
            outfile.write('# W01_means shape: {0}\n'.format(self.W01_means.shape))
            np.savetxt(outfile, self.W01_means, fmt='%-7.2f')
            outfile.write('# W12_means shape: {0}\n'.format(self.W12_means.shape))
            np.savetxt(outfile, self.W12_means, fmt='%-7.2f')
            outfile.write('# b01_means shape: {0}\n'.format(self.b01_means.shape))
            np.savetxt(outfile, self.b01_means, fmt='%-7.2f')


class MicroModule:
    def __init__(self, module_id, enought_episodes_num):
        print "created module " + str(module_id)
        self.module_id = module_id
        self.N_INPUT = 2
        self.N_HIDDEN = 3
        self.episodic_memory = EpisodicMemory(enought_episodes_num)
        self.params = ModuleParams(n_input=self.N_INPUT, n_hidden=self.N_HIDDEN)
        self.params.print_params_to_file(str(module_id) + '_init_sds')
        self.sample_from_posterior_params = None
        self.pymc3_model_object = None

    def add_episode_to_memory(self, X, Y):
        self.episodic_memory.add_new_episode(X, Y)

    def set_episodic_memory(self, X, Y):
        self.episodic_memory.clean()
        self.episodic_memory.X = X
        self.episodic_memory.Y = Y

    def try_consolidation(self, advi_iterations):
        if self.episodic_memory.contains_enough_memories():
            self.learn(advi_iterations)
            self.episodic_memory.clean() #TODO может не всегда?

    def learn(self, n_iters_advi):
        self.pymc3_model_object = None
        with pm.Model() as self.pymc3_model_object:
            ann_input = pm.Deterministic(name='input',
                                         var=theano.shared(np.array(self.episodic_memory.X)))
            ann_output = pm.Deterministic(name='output',
                                         var=theano.shared(np.array(self.episodic_memory.Y)))
            # задаем априорные распредления по параметрам нейросети
            W01 = pm.Normal('w01',
                            mu=self.params.W01_means,
                            sd=self.params.W01_sds,
                            shape=(self.N_INPUT, self.N_HIDDEN))

            W12 = pm.Normal('w12',
                            mu=self.params.W12_means,
                            sd=self.params.W12_sds,
                            shape=(self.N_HIDDEN,))

            b01 = pm.Normal('b01',
                            mu=self.params.b01_means,
                            sd=self.params.b01_sds,
                            shape=(self.N_HIDDEN,))

            # связываем эти параметры-распределения в нейросеть
            print "ann_input " + str(type(ann_input))
            act_1 = T.tanh(T.dot(ann_input, W01) + b01)  # активация скрытого слоя
            act_out = T.nnet.sigmoid(T.dot(act_1, W12)) # активация выходного нейрона

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli('my_out',
                               act_out,
                               observed=ann_output)
            v_params = pm.variational.advi(n=n_iters_advi)
            self.params.reset(v_params) # запоминаем параметры постериора, чтоб в будущем считать их приором (рекурсия)
            self.sample_from_posterior_params = pm.variational.sample_vp(v_params, draws=5000)

    def set_input_to_model(self, X, Y=None):
        if Y is None:
           Y = np.ones(X.shape[1], dtype=np.int8)  # заглушка
        self.pymc3_model_object['input'].set_value(X)
        self.pymc3_model_object['output'].set_value(Y)

    def feed_episode(self, X):
        """
        для данного эпизода модель генерирует распределение ppc (на основе хранящейся выборки из апостериорного рапсределения)
        возвращаем ответ нейросети и неуверенность в этом ответе (дисперсия ppc)
        :param episode: точка Х
        :return: ответ, неуверенность в ответе
        """
        self.set_input_to_model(X)
        ppc = pm.sample_ppc(trace=self.sample_from_posterior_params,
                                model=self.pymc3_model_object,
                                samples=500)

        smooth_prediction = ppc['my_out'].mean(axis=0)  # вероятность того, что класс = 1
        unsertainty = ppc['my_out'].std(axis=0)  # дисперсия гипотез пропорциональна "неуверенности" модели в ее "лучшей" гипотезе
        return smooth_prediction, unsertainty

    def get_unserts_and_probs_on_grid(self, grid_side):
        grid = np.mgrid[-3:3:(grid_side * 1j), -3:3:(grid_side * 1j)]
        grid_2d = grid.reshape(2, -1).T
        self.set_input_to_model(grid_2d)  # в кач-ве входных данных - узлы решетки
        # в этих узлах считаем распределения уверенности
        ppc = pm.sample_ppc(trace=self.sample_from_posterior_params,
                            model=self.pymc3_model_object,
                            samples=500)
        unsertainties = ppc['my_out'].std(axis=0)
        probabilities = ppc['my_out'].mean(axis=0)
        return grid[0], grid[1], unsertainties.reshape(grid_side, grid_side), probabilities.reshape(grid_side, grid_side)




def test():
    # протестируем работоспособность отдельного модуля на линейно разделимой бинарной классификации
    centers = [[1, 1], [-1, -1]]
    X, Y = make_blobs(n_samples=10, centers=centers, n_features=2, cluster_std=0.5,
                      random_state=0)
    plt.figure(1)
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 0')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='r', label='Class 1')
    sns.despine()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Real data')
    plt.show()
    # создадим модуль и обучим его
    module = MicroModule(module_id=666)
    module.set_episodic_memory(X, Y)
    module.learn()
    # визуализиуем его ответы
    module.visualise_model(X, Y)


if __name__ == "__main__":
    test()