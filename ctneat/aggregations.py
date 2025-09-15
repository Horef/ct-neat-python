import inspect
from typing import List, Dict, Optional, Union, Iterable

"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

import types
import warnings
from functools import reduce
from operator import mul

from ctneat.math_util import mean, median2


def product_aggregation(x: Iterable[float]) -> float:  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x: Iterable[float]) -> float:
    return sum(x)


def max_aggregation(x: Iterable[float]) -> float:
    return max(x)


def min_aggregation(x: Iterable[float]) -> float:
    return min(x)


def maxabs_aggregation(x: Iterable[float]) -> float:
    return max(x, key=abs)


def median_aggregation(x: Iterable[float]) -> float:
    return median2(x)


def mean_aggregation(x: Iterable[float]) -> float:
    return mean(x)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")

    sig = inspect.signature(function)
    params = list(sig.parameters.values())

    if len(params) == 1:
        # Assumes it is a function that takes an iterable.
        if not (params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise InvalidAggregationFunction("A function taking one argument is required")
        return function

    if len(params) == 2:
        # Assumes it is a function that takes two arguments, suitable for reduce.
        if not all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params):
            raise InvalidAggregationFunction("A function taking two arguments is required for reduce")

        def reducer(x: Iterable[float]) -> float:
            return reduce(function, x)

        return reducer

    raise InvalidAggregationFunction(f"A function taking either one or two arguments is required, "
                                     f"but {function.__name__} takes {len(params)}")


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, function):
        self.functions[name] = validate_aggregation(function)

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
