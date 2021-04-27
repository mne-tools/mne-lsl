import math
from sklearn.metrics import confusion_matrix as sk_confusion_matrix 
import numpy as np

from neurodecode import logger

#----------------------------------------------------------------------
def sigmoid(x):
    """
    Standard sigmoid
    """
    return 1 / (1 + math.exp(-x))

#----------------------------------------------------------------------
def sigmoid_array(x):
    """
    Sigmoid for a np.Array
    """
    return 1 / (1 + np.exp(-x))

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
def beta(alpha, n):
    """
    Multinomial Beta function with uniform alpha values
    
    Parameters
    ----------
    n : int
        The number of rule probabilities
    """
    return math.gamma(alpha) ** n / math.gamma(n * alpha)

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
def average_every_n(arr, n):
    """
    Average every n elements of a numpy array

    if not len(arr) % n == 0, it will be trimmed to the closest divisible length
    
    Parameters
    ----------
    arr : numpy.Array
        The array to average
    n : int
       Define the number of elements to average.
    """
    end = n * int(len(arr) / n)
    return np.mean(arr[:end].reshape(-1, n), 1)

#----------------------------------------------------------------------
def confusion_matrix(Y_true, Y_pred, label_len=6):
    """
    Generate confusion matrix in a string format

    Parameters
    ----------
    Y_true : list
        The true labels
    Y_pred : list
        The test labels
    label_len : int
        The maximum label text length displayed (minimum length: 6)

    Returns
    -------
    cfmat : str
        The confusion matrix in str format (X-axis: prediction, -axis: ground truth)
    acc : float
        The accuracy
    """

    # find labels
    if type(Y_true) == np.ndarray:
        Y_labels = np.unique(Y_true)
    else:
        Y_labels = [x for x in set(Y_true)]
    
    # Check the provided label name length
    if label_len < 6:
        label_len = 6
        logger.warning('label_len < 6. Setting to 6.')
    label_tpl = '%' + '-%ds' % label_len
    col_tpl = '%' + '-%d.2f' % label_len

    # sanity check
    if len(Y_pred) > len(Y_true):
        raise RuntimeError('Y_pred has more items than Y_true')
    elif len(Y_pred) < len(Y_true):
        Y_true = Y_true[:len(Y_pred)]

    cm = sk_confusion_matrix(Y_true, Y_pred, Y_labels)

    # compute confusion matrix
    cm_rate = cm.copy().astype('float')
    cm_sum = np.sum(cm, axis=1)

    # Fill confusion string
    for r, s in zip(cm_rate, cm_sum):
        if s > 0:
            r /= s
    cm_txt = label_tpl % 'gt\dt'
    for l in Y_labels:
        cm_txt += label_tpl % l[:label_len]
    cm_txt += '\n'
    for l, r in zip(Y_labels, cm_rate):
        cm_txt += label_tpl % l[:label_len]
        for c in r:
            cm_txt += col_tpl % c
        cm_txt += '\n'

    # compute accuracy
    correct = 0.0
    for c in range(cm.shape[0]):
        correct += cm[c][c]
    cm_sum = cm.sum()
    if cm_sum > 0:
        acc = correct / cm.sum()
    else:
        acc = 0.0

    return cm_txt, acc
