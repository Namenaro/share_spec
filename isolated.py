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
from micro_module import MicroModule

sns.set_style('white')
sns.set_context('talk')
import model_visualiser
import dispatcher

main_folder = 'ISOLATED'
filenameX = 'bufferX.txt'
filenameY = 'bufferY.txt'
visualisation = model_visualiser.Visualizer()
iter = 0

def readXY_from_file():
    X = None
    Y = None
    with file(filenameX, 'r') as infile:
        X = np.loadtxt(infile)
    with file(filenameY, 'r') as infile:
        Y = np.loadtxt(infile)
    assert X is not None and Y is not None
    return X, Y

def create_initial_file():
    data_generator = dispatcher.DataGenerator()
    X, Y = data_generator.get_n_points(n_samples=1000)
    with file(filenameX, 'w') as outfile:
        np.savetxt(outfile, X, fmt='%-7.2f')
    with file(filenameY, 'w') as outfile:
        np.savetxt(outfile, Y, fmt='%-7.2f')

def make_next_step():
    iter += 1
    directory = "IS_" + str(iter)
    if not os.path.exists(directory):
        os.makedirs(directory)
        micro_module = MicroModule(module_id=0, enought_episodes_num=2)
        # считываем обучающие данные из файла и обучаем
        X, Y = readXY_from_file()
        micro_module.set_episodic_memory(X, Y)
        micro_module.learn(n_iters_advi=3000)
        # обученную сеть визуализируем вместе с этими данными
        x1, x2, unsertainties, probabilities = micro_module.get_unserts_and_probs_on_grid(grid_side=100)
        visualisation.visualise_model(realX=X,
                                      realY=Y,
                                      x1=x1,
                                      x2=x2,
                                      unsertainties=unsertainties,
                                      probabilities=probabilities,
                                      module_id=micro_module.module_id,
                                      directory=directory)


if __name__ == "__main__":
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    else:
        files = glob.glob('/' + main_folder + '/*')
        for f in files:
            os.remove(f)
    os.chdir(main_folder)
    create_initial_file()
    X, Y = readXY_from_file()
    while True:
        raw_input("Press Enter to continue...")
        print "next iteration: " + str(iter)
        make_next_step()