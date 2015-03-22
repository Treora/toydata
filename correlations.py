from __future__ import division

import numpy as np

from helpers import argmax


def normalise_std(features):
    normfeatures = features - np.mean(features, axis=0)
    std = np.std(normfeatures, axis=0)
    std[std == 0] = 1 # prevent dividing zero by zero
    normfeatures /= std
    return normfeatures


def normalise_binary(features):
    """Normaliser that turns correlation into checking round(a)==round(b)"""
    if np.any(features > 1) or np.any(features < 0):
        raise ValueError("Features are assumed to be between 0 and 1.")
    return np.round(features)*2-1


def column_correlation(a, b, normalise=None):
    """Given two matrices, return the correlation matrix of their columns"""
    assert len(a) == len(b)
    N = len(a)
    if normalise is not None:
        a = normalise(a)
        b = normalise(b)
    return np.array(np.mat(a).T * b)/N


def order_by_column_correlation(a, b, normalise=None):
    """Given two matrices, return a permutation of columns of the second matrix
       such that its columns correlate maximally with those of the first.
       Note that absolute value of the correlation is used.
    """
    correlations = column_correlation(a, b, normalise=normalise)
    b_permutation, _ = maximally_correlating_ordering(correlations)
    new_b = b[:, b_permutation]
    # Reorder the columns of the correlation matrix accordingly
    correlations = correlations[:, b_permutation]
    return new_b, correlations


def maximally_correlating_ordering(correlations):
    """Given the correlation matrix of columns of some matrices a and b, return
       ordering indices such that b[:, ordering] correlates maximally with a.
       Maximally correlating pairs of columns are chosen greedily.
    """
    c = np.array(correlations)
    n = len(c)
    permutation = np.zeros(n, dtype=int)
    correlation_signs = np.zeros(n, dtype=int)
    # In a greedy fashion, match best correlating columns with each other
    for _ in range(n):
        # Find the best correlation of columns that are both still available
        a_col, b_col = argmax(abs(c))
        permutation[a_col] = b_col
        correlation_signs[a_col] = np.sign(c[a_col, b_col])
        # Mark found columns as unavailable
        c[a_col, :] = 0
        c[:, b_col] = 0
    return permutation, correlation_signs


def test_column_correlation_std():
    a = np.array([
        [ 3.0, 0.1],
        [ 2.0, 0.2],
        [ 1.0, 0.1],
    ])
    b = np.array([
        [ 4.0, 0.1],
        [ 7.0, 0.2],
        [ 4.0, 0.3],
    ])
    c = column_correlation(a, b, normalise_std)

    assert np.allclose(c, [
        [ 0, -1],
        [ 1,  0]
    ])


def test_column_correlation_binary():
    a = np.array([
        [ 0.4, 0.3],  # 0 0
        [ 1.0, 1.0],  # 1 1
        [ 0.1, 0.7],  # 0 1
    ])
    b = np.array([
        [ 0.6, 0.1],  # 1 0
        [ 0.4, 0.2],  # 0 0
        [ 0.6, 0.3],  # 1 0
    ])
    c = column_correlation(a, b, normalise=normalise_binary)
    assert np.allclose(c, [
        [  -1,  1/3],
        [-1/3, -1/3]
    ])


def test_order_by_column_correlation():
    a = np.array([
        [ 3.0, 0.1],
        [ 2.0, 0.2],
        [ 1.0, 0.1],
    ])
    b = np.array([
        [ 4.0, 0.1],
        [ 7.0, 0.2],
        [ 4.0, 0.3],
    ])
    new_b, correlations = order_by_column_correlation(a, b,
                                                      normalise=normalise_std)
    assert np.all(new_b == [
        [ 0.1, 4.0],
        [ 0.2, 7.0],
        [ 0.3, 4.0],
    ])


def test_maximally_correlating_ordering():
    correlations = [
        [ 0.5,  0.7, -0.3],
        [ 0.2, -0.2, -0.6],
        [ 0.7,  0.8, -0.4],
    ]
    ordering, inversions = maximally_correlating_ordering(correlations)
    # The strongest available correlations are 0.8, -0.6 and 0.5 (the 0.7
    # values are to be ignored as 0.8 is the better value in their column/row)
    assert np.all(ordering == [0, 2, 1])
    assert np.all(inversions == [1, -1, 1])
