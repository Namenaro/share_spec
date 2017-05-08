# -*- coding: utf-8 -*
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False)
ax1.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax1.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine()
ax1.legend()
ax1.set(xlabel='X', ylabel='Y', title='Toy binary classification data set')

print "dataset generated"

ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)

n_hidden = 5

# Initialize random weights between each layer
init_1 = np.random.randn(X.shape[1], n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)

print "create model..."
with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                             shape=(X.shape[1], n_hidden),
                             testval=init_1)

    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                            shape=(n_hidden, n_hidden),
                            testval=init_2)

    # Weights from hidden lay2er to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                              shape=(n_hidden,),
                              testval=init_out)

    # Build neural-network using tanh activation function
    act_1 = T.tanh(T.dot(ann_input,
                         weights_in_1))
    act_2 = T.tanh(T.dot(act_1,
                         weights_1_2))
    act_out = T.nnet.sigmoid(T.dot(act_2,
                                   weights_2_out))

    # Binary classification -> Bernoulli likelihood
    out = pm.Bernoulli('out',
                       act_out,
                       observed=ann_output)


    print "start bayesian inference on model.."
    # Run ADVI which returns posterior means, standard deviations, and the evidence lower bound (ELBO)
    v_params = pm.variational.advi(n=50000)


    trace = pm.variational.sample_vp(v_params, draws=5000)

    ax2.plot(v_params.elbo_vals)
    ax2.set(xlabel='iteration', ylabel='ELBO', title='ELBO during optimisation')


    # Replace shared variables with testing set
    ann_input.set_value(X_test)
    ann_output.set_value(Y_test)

    # Creater posterior predictive samples
    ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

    # Use probability of > 0.5 to assume prediction of class 1
    pred = ppc['out'].mean(axis=0) > 0.5

    # Replace shared variables with testing set
    ann_input.set_value(X_test)
    ann_output.set_value(Y_test)

    # Creater posterior predictive samples
    ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

    # Use probability of > 0.5 to assume prediction of class 1
    pred = ppc['out'].mean(axis=0) > 0.5


    ax3.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax3.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
    sns.despine()
    ax3.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y')

    print 'Accuracy = ' + str(((Y_test == pred).mean() * 100))
    plt.show()

    #Lets look at what the classifier has learned
    grid = np.mgrid[-3:3:100j, -3:3:100j]
    grid_2d = grid.reshape(2, -1).T
    X, Y = grid
    dummy_out = np.ones(grid.shape[1], dtype=np.int8)

    ann_input.set_value(grid_2d.astype(np.float32))
    ann_output.set_value(dummy_out)

    # Creater posterior predictive samples
    ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

    # Probability surface
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(X, Y, ppc['out'].mean(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y')
    cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')
    plt.show()

    # Uncertainty in predicted value
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(X, Y, ppc['out'].std(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y')
    cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')
    plt.show()