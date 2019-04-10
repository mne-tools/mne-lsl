from __future__ import print_function, division

"""
Regularized LDA

The code was implemented based on the following paper:
Dornhege et al., "General Signal Processing and Machine Learning Tools for
BCI Analysis Toward Brain-Computer Interfacing", MIT Press, 2007, page 218.

Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import sys
import math
import numpy as np
from pycnbi import logger

class rLDA(object):
    def __init__(self, reg_cov=None):
        if reg_cov > 1:
            raise RuntimeError('reg_cov > 1')
        self.lambdaStar = reg_cov

    def fit(self, X, Y):
        """
        Train rLDA

        Input
        -----
        X(Data): 2-D numpy array. [ samples x features ]
        Y(Label): 1-D numpy array.

        Output
        ------
        w: weight vector. [ samples ]
        b: bias scalar.

        Note that the rLDA object itself is also updated with w and b, i.e.,
        the return values can be safely ignored.

        """
        labels = np.unique(Y)
        if X.ndim != 2:
            raise RuntimeError('X must be 2 dimensional.')
        if len(labels) != 2 or labels[0] == labels[1]:
            raise RuntimeError('Exactly two different labels required.')

        index1 = np.where(Y == labels[0])[0]
        index2 = np.where(Y == labels[1])[0]
        cov = np.matrix(np.cov(X.T))
        mu1 = np.matrix(np.mean(X[index1], axis=0).T).T
        mu2 = np.matrix(np.mean(X[index2], axis=0).T).T
        mu = (mu1 + mu2) / 2;
        numFeatures = X.shape[1]

        if self.lambdaStar is not None and numFeatures > 1:
            cov = (1 - self.lambdaStar) * cov + (self.lambdaStar / numFeatures) * np.trace(cov) * np.eye(cov.shape[0])

        w = np.linalg.pinv(cov) * (mu2 - mu1)
        b = -(w.T) * mu

        for wi in w:
            assert not math.isnan(wi)
        assert not math.isnan(b)

        self.w = np.array(w).reshape(-1)  # vector
        self.b = np.array(b).reshape(-1)  # scalar
        self.labels = labels

        return self.w, self.b

    def predict(self, X, proba=False):
        """
        Returns the predicted class labels optionally with likelihoods
        """
        probs = []
        predicted = []
        for row in X:
            probability = float(self.w.T * np.matrix(row).T + self.b.T)
            if probability >= 0:
                predicted.append(self.labels[1])
            else:
                predicted.append(self.labels[0])

            # rescale from 0 to 1, similar to scikit-learn's way
            prob_norm = 1.0 / (np.exp(-probability / 10.0) + 1.0)
            # values are in the same order as that of self.labels
            probs.append([1 - prob_norm, prob_norm])

        if proba:
            return np.array(probs)
        else:
            return predicted

    def predict_proba(self, X):
        """
        Returns the predicted class labels and likelihoods
        """
        return self.predict(X, proba=True)

    def get_labels(self):
        """
        Returns labels in the same order as you get when you call predict()
        """
        return self.labels

    def score(self, X, true_labels):
        raise RuntimeError('SORRY: FUNCTION IS NOT IMPLEMENTED YET.')
