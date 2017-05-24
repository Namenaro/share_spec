# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from pymc3 import sample
from pymc3 import Slice
from pymc3 import find_MAP
from scipy import optimize
from pymc3 import summary
from pymc3 import traceplot
from pymc3 import Model, Normal, HalfNormal

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = 1.5

# Size of dataset
size = 100

# Predictor variable
X = np.random.randn(size)


# Simulate outcome variable
Y = alpha + beta*X + np.random.randn(size)*sigma

basic_model = Model()
with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10)
    sigma = HalfNormal('sigma', sd=1)

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=alpha + beta*X, sd=sigma, observed=Y)



with basic_model:
    # obtain starting values via MAP
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = Slice(vars=[sigma])

    # draw 5000 posterior samples
    trace = sample(500, step=step, start=start)

    traceplot(trace)
    plt.show()
    summary(trace)