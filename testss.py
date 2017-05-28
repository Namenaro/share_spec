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
t = np.arange(0.0, 2.0, 0.01)
s1 = 1 + np.sin(2*np.pi*t)
s2 = 1 + 2*np.sin(2*np.pi*t)
ax1.plot(t,s1)
ax2.plot(t,s2)

import os
import glob
main_folder = 'results'
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
else:
    files = glob.glob('/' + main_folder +'/*')
    for f in files:
        os.remove(f)
os.chdir(main_folder)

directory = 'iteration'
if not os.path.exists(directory):
    os.makedirs(directory)

plt.savefig(directory + "/test.png")
