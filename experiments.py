from __future__ import division, print_function

import random

import numpy as np

from dataset import generate_dataset, generate_uncorrelated_dataset, \
                    generate_single_feature_dataset
from evaluation import evaluate_model, compare
from stacked_rbms import StackedRBMs
try:
    from mlp import MLP
except ImportError:
    def MLP(*a): raise RuntimeError("MLP not available. Is Theano installed?")

from helpers import make_update_globals
from visualisation import plot_correlations, plot_weights, save_figure

update_globals = make_update_globals(globals())

N_train = 2000
N_test = 2000


def experiment_rbm_learning():
    """Train and test an RBM with normal toy data"""
    trainingset = generate_dataset(N_train)
    testset = generate_dataset(N_test)
    model = StackedRBMs(trainingset.layer_sizes)

    # Training
    train_unsupervised(model, trainingset)

    # Testing
    results = evaluate_model(model, testset)
    return results


def experiment_rbm_power():
    """Test if an RBM has the representational power to learn the desired
       features"""
    trainingset = generate_dataset(N_train)
    testset = generate_dataset(N_test)
    model = StackedRBMs(trainingset.layer_sizes)

    # Train the model using also the (not so) hidden labels
    train_with_hidden_labels(model, trainingset)

    # Testing
    results = evaluate_model(model, testset)
    return results


def experiment_rbm_power_stability():
    """Test if 'ideal' parameters for us also form an optimum for the RBMs

       We first do the test for representational power, and then continue with
       unsupervised training to see whether the parameters stay around the same
       values.
    """
    # First, train with hidden labels to test representational power
    results0 = experiment_rbm_power()
    model = results0['model']
    testset = results0['testset']

    # Then, unsupervised training to see if parameters stay or change
    trainingset1 = generate_dataset(N_train)
    train_unsupervised(model, trainingset1)
    results1 = evaluate_model(model, testset)

    return results0, results1


def experiment_mlp_learning():
    """Test if an MLP learns the desired features"""
    trainingset = generate_dataset(N_train)
    testset = generate_dataset(N_test)
    model = MLP(trainingset.layer_sizes)

    # Training
    train_supervised(model, trainingset)

    # Testing
    results = evaluate_model(model, testset)
    return results


def experiment_mlp_power():
    """Try to find 'ideal' parameters to test if the MLP has the
       representational power to required to learn the desired features.
    """
    trainingset = generate_dataset(N_train)
    testset = generate_dataset(N_test)
    model = MLP(trainingset.layer_sizes)

    # Train the model using also the (not so) hidden labels
    train_with_hidden_labels(model, trainingset)

    # Testing
    results = evaluate_model(model, testset)
    return results


def experiment_mlp_power_uncorrelated_features():
    """A variation of experiment_mlp_power.
       Train the hidden layer with an 'uncorrelated' dataset, so it can not
       cheat and learn to detect features by looking for commonly co-occurring
       features.
    """
    uncorrelated_trainingset = generate_uncorrelated_dataset(N_train)
    trainingset = generate_dataset(N_train)
    testset = generate_dataset(N_test)

    model = MLP(trainingset.layer_sizes)

    model.train_bottom(*uncorrelated_trainingset.get_layers()[:-1])
    model.train_top(*trainingset.get_layers()[1:])

    results = evaluate_model(model, testset)
    return results


def experiment_data_autocorrelation():
    """Test how the hidden labels in the data correlate among themselves"""
    testset = generate_dataset(N_train)
    ls = testset.get_layers()[1:]

    # Compare the (hidden) labels to themselves
    metrics = compare(ls, ls)

    return locals()


def train_unsupervised(model, dataset):
    """Train the model using only the visible layer of the data"""
    v = dataset.get_layer(0)
    model.train(v)


def train_supervised(model, dataset):
    """Train with visible data and the top layer of the labels"""
    v = dataset.get_layer(0)
    l_top = dataset.get_layer(-1)
    given_data = [v] + [None]*(dataset.n_layers-2) + [l_top]
    model.train(*given_data)


def train_with_hidden_labels(model, dataset):
    """Train the model super-supervised, using the visible layer and all
       (not anymore hidden) labels"""
    model.train(*dataset.get_layers())


def set_random_seed(seed=None):
    if seed==None:
        seed = random.randint(1, 1e6)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def run_experiment(experiment, show=True):
    """Run a given experiment, and plot resulting correlations and weights"""

    experiment_name = experiment.func_name
    if experiment_name.startswith('experiment_'):
        experiment_name = experiment_name[len('experiment_'):]
    print("Running {}".format(experiment_name))

    seed = set_random_seed()
    print("Using seed: {}".format(seed))

    results = experiment()

    if type(results) is dict:
        process_results(results, experiment_name, show)
    else:
        # For experiments returning multiple sets of results
        for (i, results_) in enumerate(results):
            process_results(results_, experiment_name+`i`, show)


def process_results(results, experiment_name, show):
    # Make results available for inspection when running interactively
    update_globals(results, experiment_name)

    if 'metrics' in results:
        # Plot correlations and weights
        name_pattern = 'results/' + experiment_name  + '_{}.png'
        if 'correlations' in results['metrics']:
            correlations = results['metrics']['correlations']
            fig_correlations = plot_correlations(correlations)
            save_figure(fig_correlations, name_pattern.format('correlations'))
            if show:
                fig_correlations.show()
        if 'ws' in results:
            weights = results['ws']
            fig_weights = plot_weights(weights)
            save_figure(fig_weights, name_pattern.format('weights'))
            if show:
                fig_weights.show()


def run_all_experiments():
    print("Running all experiments, dumping plots in ./results/")
    import os
    try: os.mkdir('results')
    except: pass  # probably just exists already
    all_experiments = [v for (k,v) in globals().items()
                       if k.startswith('experiment_')]
    for experiment in all_experiments:
        run_experiment(experiment, show=False)
