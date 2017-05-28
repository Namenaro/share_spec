# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')



m = {}
m[9]=6.5
m[8] = 4.8
m[10] = 3.1
m[11]=11.6
sorted_ids = sorted(m.items(), key=lambda x: x[1], reverse=True)  # сначала большие, потом маленькие
winner_id = sorted_ids[0][0]
print str(winner_id)