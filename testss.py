# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)
a = np.random.random_sample((3, 3))
print a
print 5*a

iters_init = range(100, 4000, 1000)
num_episodes_init = range(2, 20, 7)
iters_consolidation = range(1000, 10000, 3000)
num_modules = range(2, 20, 3)
enought_episodes_in_module = range(3, 15, 4)
for ii, nei, ic, nm, eeim in zip(iters_init, num_episodes_init, iters_consolidation, num_modules,
                                 enought_episodes_in_module):
    folder = "e" + str(ii) + "_" + str(nei) + "_" + str(ic) + "_" + str(nm) + "_" + str(eeim)
    print folder

m = []
m.append((1,1))
m.append((2,2))
m.append((3,1))
m.append((3,2))
m.append((4,1))
m.append((4,2))
y = [0,0,1,1,2,2]
fig, ax_prob = plt.subplots(1, 1, sharex=True)
m = np.array(m)
ax_prob.scatter(m[:,0], m[:, 1],c=y)
plt.show()
print str(np.array(m))