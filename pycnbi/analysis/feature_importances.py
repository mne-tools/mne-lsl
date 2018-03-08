"""
Analyze information content per frequency band and per channel from
Random Forest feature importance matrix data file.

Kyuhwa Lee, 2017
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
import numpy as np
import scipy.io

def feature_importances(featfile, channels, matfile=None):
    """
    input
    -----
    featfile: feature analysis file
    channels: channel names in list
    matfile: export feature importance to Matlab file if not None
    """
    # channel index lookup table
    ch2index = {ch:i for i, ch in enumerate(channels)}
    data_delta = np.zeros(len(channels))
    data_theta = np.zeros(len(channels))
    data_mu = np.zeros(len(channels))
    data_beta = np.zeros(len(channels))
    data_lgamma = np.zeros(len(channels))
    data_hgamma = np.zeros(len(channels))
    data_per_ch = np.zeros(len(channels))
    data_all = {ch:{} for ch in channels}

    f = open(featfile)
    f.readline()
    for l in f:
        token = l.strip().split('\t')
        importance = float(token[0])
        ch = token[1]
        fq = float(token[2])
        data_all[ch][fq] = importance
        if fq <= 3:
            data_delta[ch2index[ch]] += importance
        elif fq <= 7:
            data_theta[ch2index[ch]] += importance
        elif fq <= 12:
            data_mu[ch2index[ch]] += importance
        elif fq <= 30:
            data_beta[ch2index[ch]] += importance
        elif fq <= 70:
            data_lgamma[ch2index[ch]] += importance
        else:
            data_hgamma[ch2index[ch]] += importance
        data_per_ch[ch2index[ch]] += importance

    print('>> Feature importance distribution')
    if matfile is not None:
        # matvar = [fq] x [ch]
        matvar = np.zeros([len(data_all[channels[0]]), len(channels)])
        fqlist = sorted(list(data_all[channels[0]].keys()))
        fq2index = {fq:i for i, fq in enumerate(fqlist)}
        for ch in channels:
            for fq in fqlist:
                matvar[fq2index[fq]][ch2index[ch]] = data_all[ch][fq]
        scipy.io.savemat(matfile, {'fid':matvar, 'channels':channels, 'frequencies':fqlist})

    print('bands  ', qc.list2string(channels, '%6s'), '|', 'per band')
    print('-' * 66)
    print('delta  ', qc.list2string(data_delta, '%6.2f'), '| %6.2f' % np.sum(data_delta))
    print('theta  ', qc.list2string(data_theta, '%6.2f'), '| %6.2f' % np.sum(data_theta))
    print('mu     ', qc.list2string(data_mu, '%6.2f'), '| %6.2f' % np.sum(data_mu))
    print('beta   ', qc.list2string(data_beta, '%6.2f'), '| %6.2f' % np.sum(data_beta))
    print('lgamma ', qc.list2string(data_lgamma, '%6.2f'), '| %6.2f' % np.sum(data_lgamma))
    print('hgamma ', qc.list2string(data_hgamma, '%6.2f'), '| %6.2f' % np.sum(data_hgamma))
    print('-' * 66)
    print('per_ch ', qc.list2string(data_per_ch, '%6.2f'), '| 100.00' )

if __name__ == '__main__':
    featfile = r'D:\data\MI\z2\LR\classifier\good_features.txt'
    channels = ['F3','F4','C3','Cz','C4','P3','Pz','P4']
    feature_importances(featfile, channels)
