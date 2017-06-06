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