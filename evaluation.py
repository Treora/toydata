from __future__ import division

import numpy as np

from correlations import column_correlation, maximally_correlating_ordering, \
                         normalise_binary


def evaluate_model(model, testset):
    """Evaluate how well the model infers the (hidden) labels of test data"""

    # Sort data by top level label to ease inspection
    testset = testset.sort_using_layer(-1, reverse=True)

    # Feed the samples to the model to obtain each layers' activations
    v = testset.get_layer(0)
    hs = model.transform(v)[1:]

    # Read model weights
    ws = [params['w'] for params in model.parameters]
    del params

    # Take the (hidden) labels from the data set
    ls = testset.get_layers()[1:]

    # In each layer, reorder and invert neurons to match best with the labels
    for i in range(len(ls)):
        hs[i], ws[i] = align_with_labels(ls[i], hs[i], ws[i])
    del i

    # Measure correlations, etcetera
    metrics = compare(ls, hs)

    # Simply return a dict with all used variables
    return locals()


def align_with_labels(l, h, w):
    # Compare the unordered activations to the (hidden) labels
    correlations = column_correlation(l, h, normalise=normalise_binary)
    ordering, inversions = maximally_correlating_ordering(correlations)

    inversions = np.reshape(inversions, (1, -1)) # make it a 2d array
    # Reorder and set h:=1-h when a match was negatively correlating
    new_h = 0.5 + inversions * (-0.5 + h[:, ordering])
    # Reorder and negate rows of weight matrices accordingly
    new_w = inversions.T * w[ordering, :]

    return new_h, new_w


def compare(ls, hs):
    # Compare the activations to the (hidden) labels (also compare cross-layer)
    correlations = [
        [column_correlation(l, h, normalise=normalise_binary) for h in hs]
        for l in ls
    ]
    del l, h

    # Score the overall 'correctness' of activations using average correlation
    correctnesses = [
        np.trace(layer_correlations)/len(layer_correlations)
        for layer_correlations in np.diagonal(correlations)
    ]
    del layer_correlations

    # Measure how well the model would classify the top level labels.
    class_predictions = hs[-1].argmax(axis=1)
    actual_classes = ls[-1].argmax(axis=1)
    class_accuracy = ( np.count_nonzero(class_predictions == actual_classes)
                       / len(class_predictions) )

    return locals()
