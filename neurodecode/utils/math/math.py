import math
import numpy as np


def sigmoid(x):
    """
    Standard sigmoid
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    return 1 / (1 + np.exp(-x))


def dirichlet(n):
    """
    Uniform Dirichlet distribution with sigma(alpha)=1.0

    Parameters
    ----------
    n : int
        The number of rule probabilities
    """
    alpha = 1.0 / n

    return 1 / beta(alpha, n) * ((1 / n) ** (alpha - 1)) ** n


def beta(alpha, n):
    """
    Multinomial Beta function with uniform alpha values.

    Parameters
    ----------
    n : int
        The number of rule probabilities
    """
    return math.gamma(alpha) ** n / math.gamma(n * alpha)


def poisson(mean, k):
    """
    Poisson distribution. We use k-1 since the minimum length is 1, not 0.

    Parameters
    ----------
    mean : float
        The poisson rate
    k : int
        The poisson support
    """
    return (mean ** (k - 1) * math.exp(-mean)) / math.factorial(k - 1)


def average_every_n(arr, n):
    """
    Average every n elements of a numpy array.

    if not len(arr) % n == 0, it will be trimmed to the closest divisible
    length.

    Parameters
    ----------
    arr : numpy.Array
        The array to average
    n : int
       Define the number of elements to average.
    """
    end = n * int(len(arr) / n)
    return np.mean(arr[:end].reshape(-1, n), 1)
