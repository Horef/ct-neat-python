"""
Commonly used functions not available in the Python2 standard library.
Currently only maintained for backwards compatibility.
"""

from math import sqrt
import numpy as np


def mean(values):
    values = list(values)
    return np.mean(values).item()


def median(values):
    values = list(values)
    return np.median(values).item()


def median2(values):
    values = list(values)
    return np.median(values).item()


def variance(values):
    values = list(values)
    return np.var(values).item()


def stdev(values):
    values = list(values)
    return np.std(values).item()


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..)."""
    values = list(values)
    x = np.asarray(values)
    exps = np.exp(x - np.max(x))
    return (exps / np.sum(exps)).tolist()


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2}
