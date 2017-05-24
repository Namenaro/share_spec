# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import numpy as np

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

    def try_consolidation(self):
        if self.episodic_memory.contains_enough_memories():
            self.learn()

    def learn(self):
        self.pymc3_model_object = None
        ann_input = theano.shared(self.episodic_memory.X)
        ann_output = theano.shared(self.episodic_memory.Y)
        with pm.Model() as neural_network:
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
            v_params = pm.variational.advi(n=50000)
            self.params.reset(v_params)
            self.sample_from_posterior_params = pm.variational.sample_vp(v_params, draws=5000)
        self.pymc3_model_object = neural_network

    def feed_episode(self, X, Y):
        """
        для данного эпизода модель генерирует распределение ppc (на основе хранящейся выборки из апостериорного рапсределения)
        возвращаем ответ нейросети и неуверенность в этом ответе (дисперсия ppc)
        :param episode: точка Х
        :return: ответ, неуверенность в ответе
        """
        with self.pymc3_model_object:
            ann_input.set_value(X)
            ann_output.set_value(Y)

            ppc = pm.sample_ppc(trace=self.sample_from_posterior_params,
                                model=self.pymc3_model_object,
                                samples=500)

            # Если матожидание выборки > 0.5 то класс 1, иначе класс 0
            prediction = ppc['my_out'].mean(axis=0) > 0.5

