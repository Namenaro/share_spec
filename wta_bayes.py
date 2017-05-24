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
import matplotlib as mpl
import os
import glob

files = glob.glob('/results/*')
for f in files:
    os.remove(f)
os.chdir('results')

cmap = mpl.cm.autumn

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

def show_only_posterior_mu(v_params, some_str, my_color):
    x = np.linspace(-10, 10, 100)
    y = stats.norm(v_params.means['mu'], v_params.stds['mu']).pdf(x)
    plt.plot(x, y, color=cmap(my_color))
    pylab.savefig('mu_'+some_str + '.png')

def get_data():
    return np.random.normal(size=3, loc=2.)

current_mean_my_mu=-3.0
current_mean_my_sd = 12
plt.figure(figsize=(8,2))
plt.ylim(ymax=1.01, ymin=0.0)
plt.title('mu')
plt.ylabel('probability density')
plt.axvline(x=2.0, label='real mu')
plt.legend(loc=0)
num_iterations = 25
for iteration in range(num_iterations):
    data = get_data()
    iter_name = "iter_" + str(iteration)
    with pm.Model() as model:
        my_mu = pm.Normal('mu', mu=current_mean_my_mu, sd=current_mean_my_sd)
        my_sd = pm.HalfNormal('sd', sd=10)
        n = pm.Normal(iter_name, mu=my_mu, sd=my_sd, observed=data)
        v_params = pm.variational.advi(model=model, n=100)
        current_mean_my_mu = v_params.means['mu']
        current_mean_my_sd = v_params.stds['mu']
        color = float(num_iterations - iteration)/float(num_iterations)
        show_only_posterior_mu(v_params, iter_name, color)
