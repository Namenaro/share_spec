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

data = np.random.randn(100)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1, testval=0)
    sd = pm.HalfNormal('sd', sd=1)

show_sample(mu, "mu_prior")
show_sample(sd, "sd_prior")

with model:
    n = pm.Normal('n', mu=mu, sd=sd, observed=data)
    v_params = pm.variational.advi(model=model, n=100)

ax = sns.distplot(data, label='data')
xlim = ax.get_xlim()
x = np.linspace(xlim[0], xlim[1], 100)
y = stats.norm(v_params.means['mu'], v_params.stds['mu']).pdf(x)
ax.plot(x, y, label='ADVI')
ax.set_title('mu')
ax.legend(loc=0)
pylab.savefig('foo.png')
plt.show()