# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import seaborn as sns
sns.set_style('white')

import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self):
        pass

    def visualise_unsertainty(self, ax, x1, x2, ppc_for_x):
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        contour = ax.contourf(x1, x2, ppc_for_x['my_out'].std(axis=0).reshape(100, 100), cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')


    def visualise_propbability(self, ax, x1, x2, ppc_for_x):
        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
        contour = ax.contourf(x1, x2, ppc_for_x['my_out'].mean(axis=0).reshape(100, 100), cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')




