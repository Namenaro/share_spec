# -*- coding: utf-8 -*

from pprint import pprint

import pymc3 as pm

import numpy as np
from scipy import stats
import pymc3 as pm
import pylab
import seaborn as sns
import matplotlib.pyplot as plt

def show_sample(rv, name_str):
    samples = rv.random(size=8000)
    plt.hist(samples, bins=70, normed=True, histtype='stepfilled')
    plt.title(name_str)
    pylab.savefig(name_str + '.png')
    plt.show()

def show_data(data):
    plt.hist(data, bins=70, normed=True, histtype='stepfilled')
    plt.title("data")
    pylab.savefig("data" + '.png')
    plt.show()

data = np.random.normal(size= 10, loc=2.)
show_data(data)

with pm.Model() as model:
    my_mu = pm.Normal('mu', mu=0, sd=20)
    my_sd = pm.HalfNormal('sd', sd=10)

show_sample(my_mu, "mu_prior")
show_sample(my_sd, "sd_prior")

with model:
    n = pm.Normal('n', mu=my_mu, sd=my_sd, observed=data)
    v_params = pm.variational.advi(model=model, n=10000)

ax = sns.distplot(data, label='data')
xlim = ax.get_xlim()
x = np.linspace(xlim[0], xlim[1], 100)
y = stats.norm(v_params.means['mu'], v_params.stds['mu']).pdf(x)
ax.plot(x, y, label='ADVI')
ax.set_title('mu')
ax.legend(loc=0)
pylab.savefig('foo.png')
plt.show()