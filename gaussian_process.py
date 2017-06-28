# -*- coding: utf-8 -*
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from sklearn.datasets.samples_generator import make_moons

X, Y = make_moons(noise=0.2, random_state=0, n_samples=15)
with pm.Model() as GP:
    l = np.array([0.2, 1.0])
    K = pm.gp.cov.ExpQuad(input_dim=2, lengthscales=l)
    s = pm.HalfCauchy('s', 2.5)
    Sigma = K(X) + tt.eye(X.shape[0]) * s ** 2
    y_obs = pm.MvNormal('y_obs', mu=0, cov=Sigma, observed=Y)
    print "here:"
    trace = pm.sample(2000)
    grid_side = 100
    grid = np.mgrid[-3:3:(grid_side * 1j), -3:3:(grid_side * 1j)]
    grid_2d = grid.reshape(2, -1).T
    gp_samples = pm.gp.sample_gp(trace, y_obs, grid_2d, samples=50, random_seed=42)
