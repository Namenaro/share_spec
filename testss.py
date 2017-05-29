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
m = np.zeros(shape=(0,2))
m= np.append(m, [[1,2]], axis=0)
m= np.append(m, [[1,2]], axis=0)
print str(np.array(m))