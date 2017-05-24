# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)
data = np.random.randn(1280)
fig, (ax1, ax2)= plt.subplots(1, 2, sharex=True)
sns.distplot(data, kde=False, norm_hist=True, ax=ax1)
sns.distplot(data, kde=False, norm_hist=False, ax=ax2)
_ = ax1.set(title='Histogram of observed data(normed to 1)', xlabel='x', ylabel='# observations')
_ = ax2.set(title='Histogram of observed data', xlabel='x', ylabel='# observations')
plt.show()

mu_current = 1.
proposal_width = 0.5
proposal = norm(mu_current, proposal_width).rvs(1)
print proposal