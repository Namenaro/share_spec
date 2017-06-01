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

    def visualise_specialisations(self, realX, realY, x1, x2, arr_probabilities, arr_unsertainties, unsertainty_threshold, directory):
        num_modules = len(arr_probabilities)
        assert len(arr_probabilities) == len(arr_unsertainties)
        assert x1.shape == x2.shape
        fig, (ax_unsert, ax_prob) = plt.subplots(1, 2, sharex=True)
        result_prob = np.zeros(x1.shape)
        # для всех модулей запишем те их ответы, в которых они уверены
        for n in range(num_modules):
            for (i, j), value in np.ndenumerate(x1):
                answer_of_module = arr_probabilities[n][i, j]
                unsertainty_of_module = arr_unsertainties[n][i, j]
                if unsertainty_of_module > unsertainty_threshold:
                    if result_prob[i, j] < answer_of_module:
                        result_prob[i, j] = answer_of_module
        #отрисовка ответа ансамбля
        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
        contour = ax_prob.contourf(x1, x2, result_prob, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax_prob)
        cbar.ax.set_ylabel('answers of ensemble')
        #отрисовка уверенности ансамбля
        result_unsert = np.zeros(x1.shape)
        for n in range(num_modules):
            for (i, j), value in np.ndenumerate(x1):
                unsertainty_of_module = arr_unsertainties[n][i, j]
                if result_unsert[i, j] > unsertainty_of_module:
                    result_unsert[i, j] = unsertainty_of_module
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        contour = ax_unsert.contourf(x1, x2, result_unsert, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax_unsert)
        cbar.ax.set_ylabel('Uncertainty of ensemble')
        # для сравнения покажем реальный датасет
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax_prob)
        self._scatter_real_dataset(realX=realX, realY=realY, ax=ax_unsert)
        if directory is not None:
            plt.savefig(directory + "/" + "all" + ".png")
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