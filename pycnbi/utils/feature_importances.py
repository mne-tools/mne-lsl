import pycnbi
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
import numpy as np
from IPython import embed

def feature_importances(featfile, channels):
    # channel index lookup table
    ch2index = {ch:i for i, ch in enumerate(channels)}
    
    data_delta = np.zeros(len(channels))
    data_theta = np.zeros(len(channels))
    data_mu = np.zeros(len(channels))
    data_beta = np.zeros(len(channels))
    data_lgamma = np.zeros(len(channels))
    data_hgamma = np.zeros(len(channels))
    data_per_ch = np.zeros(len(channels))
    
    f = open(featfile)
    f.readline()
    for l in f:
        token = l.strip().split('\t')
        importance = float(token[0])
        ch = token[1]
        fq = float(token[2])
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

# sample code
if __name__ == '__main__':
    featfile = r'D:\data\CHUV\ECoG17\20171005\fif_corrected\classifier_steps_1s_1-150Hz\good_features.txt'
    channels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
    feature_importances(featfile, channels)
