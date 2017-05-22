# -*- coding: utf-8 -*

from pprint import pprint

import pymc3 as pm

import numpy as np
from scipy import stats
import pymc3 as pm
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir('results')

def show_sample(rv, name_str):
    samples = rv.random(size=8000)
    plt.hist(samples, bins=70, normed=True, histtype='stepfilled')
    plt.title(name_str)
    pylab.savefig(name_str + '.png')
    plt.show()

def show_data(data):
    plt.hist(data, bins=10, normed=True, histtype='stepfilled')
    plt.title("data")
    pylab.savefig("data" + '.png')
    plt.show()

def show_posterior(v_params, data, some_str):
    ax = sns.distplot(data, label='data')
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 100)
    y = stats.norm(v_params.means['mu'], v_params.stds['mu']).pdf(x)
    ax.plot(x, y, label='ADVI')
    ax.set_title('mu')
    ax.legend(loc=0)
    pylab.savefig('mu_'+some_str + '.png')
   # plt.show()

def get_data():
    return np.random.normal(size=10, loc=2.)

with pm.Model() as model:
    my_mu = pm.Normal('mu', mu=0, sd=120)
    my_sd = pm.HalfNormal('sd', sd=10)

show_sample(my_mu, "mu_prior")
show_sample(my_sd, "sd_prior")

test_val=0
for n in range(10):
    data = get_data()
    iter_name = "iter_" + str(n)
    with model:
        my_mu.testval = test_val
        n = pm.Normal(iter_name, mu=my_mu, sd=my_sd, observed=data)
        v_params = pm.variational.advi(model=model, n=2000)
        show_posterior(v_params, data, iter_name)
        test_val = v_params.means['mu']