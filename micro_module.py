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
    def __init__(self):
        self.MAX_EPISODES = 5
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
        del self.X
        del self.Y


class ModuleParams:
    """
    Параметры, для которых строятся апостериорные распределения
    """
    def __init__(self, n_input, n_hidden=5):
        self.W01_means = np.random.randn(n_input, n_hidden)
        self.W01_sds = np.random.randn(n_input, n_hidden)
        self.W12_means = np.random.randn(n_hidden)
        self.W12_sds = np.random.randn(n_hidden)

    def reset(self, v_params):
        self.W01_means = v_params.means['w01']
        self.W01_sds = v_params.stds['w01']
        self.W12_means = v_params.means['w12']
        self.W12_sds = v_params.stds['w12']

class MicroModule:
    def __init__(self, module_id):
        self.module_id = module_id
        self.N_INPUT = 2
        self.N_HIDDEN = 2
        self.episodic_memory = EpisodicMemory()
        self.params = ModuleParams(n_input=self.N_INPUT, n_hidden=self.N_HIDDEN)
        self.sample_from_posterior_params = None
        self.pymc3_model_object = None

    def add_episode_to_memory(self, X, Y):
        self.episodic_memory.add_new_episode(X, Y)

    def set_episodic_memory(self, X, Y):
        self.episodic_memory.clean()
        self.episodic_memory.X = X
        self.episodic_memory.Y = Y

    def try_consolidation(self):
        if self.episodic_memory.contains_enough_memories():
            self.learn()
            self.episodic_memory.clean() #TODO может не всегда?

    def learn(self):
        self.pymc3_model_object = None
        with pm.Model() as self.pymc3_model_object:
            ann_input = pm.Deterministic(name='input',
                                         var=theano.shared(self.episodic_memory.X))
            ann_output = pm.Deterministic(name='output',
                                         var=theano.shared(self.episodic_memory.Y))
            # задаем априорные распредления по параметрам нейросети
            W01 = pm.Normal('w01',
                            mu=self.params.W01_means,
                            sd=self.params.W01_sds,
                            shape=(self.N_INPUT, self.N_HIDDEN))

            W12 = pm.Normal('w12',
                            mu=self.params.W12_means,
                            sd=self.params.W12_sds,
                            shape=(self.N_HIDDEN,))
            # связываем эти параметры-распределения в нейросеть
            act_1 = T.tanh(T.dot(ann_input, W01))  # активация скрытого слоя
            act_out = T.nnet.sigmoid(T.dot(act_1, W12)) # активация выходного нейрона

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli('my_out',
                               act_out,
                               observed=ann_output)
            v_params = pm.variational.advi(n=10000)
            self.params.reset(v_params) # запоминаем параметры постериора, чтоб в будущем считать их приором (рекурсия)
            self.sample_from_posterior_params = pm.variational.sample_vp(v_params, draws=5000)

    def set_input_to_model(self, X, Y=None):
        if Y is None:
           Y = np.ones(X.shape[1], dtype=np.int8)  # заглушка
        self.pymc3_model_object['input'].set_value(X)
        self.pymc3_model_object['output'].set_value(Y)

    def feed_episode(self, X, Y):
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

    def visualise_model(self, realX, realY, folder_name):
        nx = 100
        ny = 100
        grid = np.mgrid[-3:3:(nx * 1j), -3:3:(ny * 1j)]
        grid_2d = grid.reshape(2, -1).T
        self.set_input_to_model(grid_2d) # в кач-ве входных данных - узлы решетки
        # в этих узлах считаем распределения уверенности
        ppc = pm.sample_ppc(trace=self.sample_from_posterior_params,
                            model=self.pymc3_model_object,
                            samples=500)
        visualizer = Visualizer()
        ax = visualizer.visualise_propbability(grid[0], grid[1], ppc)
        ax.scatter(realX[realY == 0, 0], realX[realY == 0, 1])
        ax.scatter(realX[realY == 1, 0], realX[realY == 1, 1], color='r')
        plt.show()

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