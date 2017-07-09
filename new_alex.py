# -*- coding: utf-8 -*
import theano
import theano.tensor
import lasagne
floatX = theano.config.floatX

from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy, squared_error
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import matplotlib.pyplot as plt
import itertools
import lasagne.layers


from sklearn.datasets.samples_generator import make_blobs, make_moons
from sklearn.preprocessing import scale
import numpy as np
import alex_data

rng = np.random.RandomState(0)
