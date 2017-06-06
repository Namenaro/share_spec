# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import seaborn as sns
sns.set_style('white')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    def __init__(self):
        pass

    def _visualise_unsertainty(self, ax, x1, x2, unsertainties):
        print "x1  has shape" + str(x1.shape)
        print "x2  has shape" + str(x2.shape)
        print "unsertainties has shape" + str(unsertainties.shape)
        assert x1.shape == x2.shape and x2.shape == unsertainties.shape
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        contour = ax.contourf(x1, x2, unsertainties, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')

    def _visualise_propbability(self, ax, x1, x2, probabilities):
        assert x1.shape == x2.shape and x2.shape == probabilities.shape
        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
        contour = ax.contourf(x1, x2, probabilities, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')

    def _scatter_real_dataset(self, realX, realY, ax):
        ax.scatter(realX[realY == 0, 0], realX[realY == 0, 1])
        ax.scatter(realX[realY == 1, 0], realX[realY == 1, 1], color='r')

    def visualise_ensemle_unsertainty(self, realX, realY, x1, x2, arr_unsertainties, directory):
        num_modules = len(arr_unsertainties)
        assert x1.shape == x2.shape
        result = np.full(x1.shape, 1.0)
        # записываем наименьшие (т.е. самые хорошие) значения неуверенностей
        for n in range(num_modules):
            for (i, j), value in np.ndenumerate(x1):
                unsertainty_of_module = arr_unsertainties[n][i, j]
                if result[i, j] > unsertainty_of_module:
                    result[i, j] = unsertainty_of_module
        fig, ax = plt.subplots(1, 1, sharex=True)
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        contour = ax.contourf(x1, x2, result, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Uncertainty of ensemble')
        # для сравнения покажем реальный датасет
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax)
        if directory is not None:
            plt.savefig(directory + "/" + "all_unsert" + ".png")
        else:
            plt.show()

    def visualise_ensemble_probability(self, realX, realY, x1, x2, arr_probabilities, arr_unsertainties, unsertainty_threshold, directory):
        num_modules = len(arr_probabilities)
        assert len(arr_probabilities) == len(arr_unsertainties)
        assert x1.shape == x2.shape
        fig, (ax_prob, ax_mods, ax_unsert) = plt.subplots(1, 3, sharex=True, sharey=True)
        result_prob = np.full(x1.shape, 0.5)
        specalized_Xs = []
        specalized_ids = []
        unspecialized_Xs = []
        unserts = np.full(x1.shape, 1.0)
        # для каждого пиксела входного пространства посмотрим, у какого модуля наибольшая уверенность
        # в этом месте, и для из этого модуля возьмем значение вероятности
        for (i, j), value in np.ndenumerate(x1):
            best_unsertainty = 1.0
            module_winner = -1
            for n in range(num_modules):
                unsertainty_of_module = arr_unsertainties[n][i, j]
                if unsertainty_of_module < best_unsertainty:
                    best_unsertainty = unsertainty_of_module
                    module_winner = n
            unserts[i, j] = best_unsertainty
            no_winner = True
            if module_winner > 0:
                if arr_unsertainties[module_winner][i, j] < unsertainty_threshold:
                    no_winner = False
                    result_prob[i, j] = arr_probabilities[module_winner][i, j]
                    specalized_Xs.append((x1[i, j], x2[i, j]))
                    specalized_ids.append(module_winner)
            if no_winner is True:
                unspecialized_Xs.append((x1[i, j], x2[i, j]))
        #отрисовка ответа ансамбля
        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
        contour = ax_prob.contourf(x1, x2, result_prob, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax_prob)
        cbar.ax.set_ylabel('answers of ensemble')
        # заштрихуем неуверенные точки
        unspecialized_Xs = np.array(unspecialized_Xs)
        ax_prob.scatter(unspecialized_Xs[:,0], unspecialized_Xs[:, 1], color='g')
        #покажем специализации модулей
        contour = ax_mods.contourf(x1, x2, result_prob, cmap=cmap)
        specalized_Xs = np.array(specalized_Xs)
        ax_mods.scatter(specalized_Xs[:, 0], specalized_Xs[:, 1], c=specalized_ids)
        # покажем карту уверенности ансамбля
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        contour = ax_unsert.contourf(x1, x2, unserts, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax_unsert)
        cbar.ax.set_ylabel('Uncertainty of ensemble')
        # для сравнения покажем реальный датасет
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax_prob)
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax_unsert)
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax_mods)
        if directory is not None:
            plt.savefig(directory + "/" + "all_prob" + ".png")
        else:
            plt.show()

    def visualise_model(self, realX, realY, x1, x2, unsertainties, probabilities, module_id, directory=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        self._visualise_propbability(ax1, x1, x2, probabilities)
        self._visualise_unsertainty(ax2, x1, x2, unsertainties)
        # для сравнения покажем реальный датасет
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax1)
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax2)
        if directory is not None:
            plt.savefig(directory + "/" + str(module_id) + ".png")
        else:
            plt.show()