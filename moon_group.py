# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random
import seaborn as sns
from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')
import model_visualiser
import dispatcher

# чисто для сравнения покажем и настоящее распределение,
# когда будем отрисовывать результаты обучения модели
data_gen = dispatcher.DataGenerator()
realX, realY = data_gen.get_n_points(100)

class ExperimentSettings:
    def __init__(self, iters_init, iters_consolidation, num_modules, enought_episodes_in_module, num_episodes_init):
        self.iters_init = iters_init
        self.num_episodes_init = num_episodes_init
        self.iters_consolidation = iters_consolidation
        self.num_modules = num_modules
        self.enought_episodes_in_module = enought_episodes_in_module

dummy_settings = ExperimentSettings(iters_init=100,
                                    iters_consolidation=100,
                                    num_modules=2,
                                    enought_episodes_in_module=2,
                                    num_episodes_init=2)

normal_settings = ExperimentSettings(iters_init=100,
                                    iters_consolidation=2000,
                                    num_modules=3,
                                    enought_episodes_in_module=2,
                                    num_episodes_init=2)


def experiment_1(folder_for_results, settings):
    visualisation = model_visualiser.Visualizer()
    my_dispatcher = dispatcher.Dispatcher(num_modules=settings.num_modules,
                                          enought_episodes_num=settings.enought_episodes_in_module)
    my_dispatcher.setup_folder_for_results(folder_for_results)
    random.seed(43)
    my_dispatcher.simple_initialisation(advi_iterations=settings.iters_init,
                                        n_samples=settings.num_episodes_init)

    for i in range(200):
        print "iteration: " + str(i)
        my_dispatcher.feed_next_data_point_to_modules()
        my_dispatcher.try_consolidation(advi_iterations=settings.iters_consolidation)
        if i % 2 == 0:
            directory = 'iter_' + str(i)
            if not os.path.exists(directory):
                os.makedirs(directory)
            arr_propbabilities = []
            arr_unsertainties = []
            # визуализируем результаты каждого эл-та ансамбля по-отдельности
            for key, micro_module in my_dispatcher.modules.items():
                micro_module.params.print_params_to_file(str(key) + '_iter_'+ str(i))
                x1, x2, unsertainties, probabilities = micro_module.get_unserts_and_probs_on_grid(grid_side=100)
                visualisation.visualise_model(realX=realX,
                                              realY=realY,
                                              x1=x1,
                                              x2=x2,
                                              unsertainties=unsertainties,
                                              probabilities=probabilities,
                                              module_id=micro_module.module_id,
                                              directory=directory)
                arr_propbabilities.append(probabilities)
                arr_unsertainties.append(unsertainties)
            # визуализируем совместные результаты ансамбля
            visualisation.visualise_ensemle_unsertainty(realX=realX,
                                                    realY=realY,
                                                    x1=x1,
                                                    x2=x2,
                                                    arr_unsertainties=arr_unsertainties,
                                                    directory=directory)
            visualisation.visualise_ensemble_probability(realX=realX,
                                                    realY=realY,
                                                    x1=x1,
                                                    x2=x2,
                                                    arr_unsertainties=arr_unsertainties,
                                                    arr_probabilities=arr_propbabilities,
                                                    unsertainty_threshold=0.6,
                                                    directory=directory)
# for data, color, group in zip(data, colors, groups):
def experiment_2():
    iters_init = range(100, 4000, 1000)
    num_episodes_init = range(2, 20, 7)
    iters_consolidation = range( 1000, 10000, 3000)
    num_modules = range(2, 20, 3)
    enought_episodes_in_module = range(3, 15, 4)
    for ii,nei,ic,nm,eeim in zip(iters_init, num_episodes_init, iters_consolidation, num_modules, enought_episodes_in_module):
        settings = ExperimentSettings(iters_init=ii,
                                      iters_consolidation=ic,
                                      num_modules=nm,
                                      enought_episodes_in_module=eeim,
                                      num_episodes_init=nei)
        folder = "e" + str(ii)+ "_" + str(nei) + "_" + str(ic) + "_" + str(nm) + "_" + str(eeim)
        experiment_1(folder, settings)

if __name__ == "__main__":
    experiment_1('ens_3hardsigmoid', normal_settings)
    #experiment_2()