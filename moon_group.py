# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy.stats import norm
sns.set_style('white')
sns.set_context('talk')
import model_visualiser
import dispatcher
import random
# чисто для сравнения покажем и настоящее распределение,
# когда будем отрисовывать результаты обучения модели
data_gen = dispatcher.DataGenerator()
realX, realY = data_gen.get_n_points(100)

def experiment_1(folder_for_results):
    my_dispatcher = dispatcher.Dispatcher(num_modules=4)
    my_dispatcher.setup_folder_for_results(folder_for_results)
    random.seed(43)
    my_dispatcher.simple_initialisation()

    for i in range(100):
        print "iteration: " + str(i)
        my_dispatcher.feed_next_data_point_to_modules()
        my_dispatcher.try_consolidation()
        if i%4 == 0:
            directory = 'iter_' + str(i)
            if not os.path.exists(directory):
                os.makedirs(directory)
            for key, micro_module in my_dispatcher.modules.items():
                micro_module.visualise_model(realX, realY, directory)

if __name__ == "__main__":
    experiment_1('results7')
