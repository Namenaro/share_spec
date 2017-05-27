# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons




class Visualizer:
    def __init__(self):
        pass

    def visualise_model(self, my_model):
        pass

    def example_contours(self):
        from sklearn.datasets.samples_generator import make_blobs
        import matplotlib.pyplot as plt
        import seaborn as sns
        import math
        sns.set_style('white')
        nx = 100
        ny = 100
        grid = np.mgrid[-3:3:(nx * 1j), -3:3:(ny * 1j)]
        grid_2d = grid.reshape(2, -1).T
        z = np.zeros(ny * nx)
        for i in range(len(grid_2d)):
            point = grid_2d[i]
            z[i] = math.sin(point[0] * point[1])
        z = z.reshape(nx, ny)

        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        fig, ax = plt.subplots()
        contour = ax.contourf(grid[0], grid[1], z, cmap=cmap)
        cbar = plt.colorbar(contour, ax=ax)
        plt.show()