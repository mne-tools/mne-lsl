"""
Export feature importance distribution using topography map

TODO: merge with parse_features.py to use its API.

"""

import os
import sys
import mne
import pycnbi
import numpy as np
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
import matplotlib.pyplot as plt
from pycnbi.pycnbi_config import PYCNBI_ROOT
from pycnbi import logger
from matplotlib.figure import Figure
from builtins import input

def export_topo(data, pos, pngfile, xlabel='', vmin=None, vmax=None, chan_vis=None, res=64, contours=0):
    mne.viz.plot_topomap(data, pos, names=chan_vis, show_names=True, res=res, contours=contours, show=False)
    plt.suptitle(xlabel)
    plt.savefig(pngfile)
    logger.info('Exported %s' % pngfile)

def feature_importances_topo(featfile, topo_layout_file=None, channels=None, channel_name_show=None):
    """
    Compute feature importances across frequency bands and channels

    @params
    topo_laytout_file: if not None, topography map images will be generated and saved.
    channel_name_show: list of channel names to show on topography map.

    """
    logger.info('Loading %s' % featfile)

    if channels is None:
        channel_set = set()
        with open(featfile) as f:
            f.readline()
            for l in f:
                ch = l.strip().split('\t')[1]
                channel_set.add(ch)
        channels = list(channel_set)

    # channel index lookup table
    ch2index = {ch:i for i, ch in enumerate(channels)}

    data_delta = np.zeros(len(channels))
    data_theta = np.zeros(len(channels))
    data_mu = np.zeros(len(channels))
    data_beta = np.zeros(len(channels))
    data_beta1 = np.zeros(len(channels))
    data_beta2 = np.zeros(len(channels))
    data_beta3 = np.zeros(len(channels))
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
        if 12.5 <= fq <= 16:
            data_beta1[ch2index[ch]] += importance
        elif fq <= 20:
            data_beta2[ch2index[ch]] += importance
        elif fq <= 28:
            data_beta3[ch2index[ch]] += importance
        data_per_ch[ch2index[ch]] += importance

    hlen = 18 + len(channels) * 7
    result = '>> Feature importance distribution\n'
    result += 'bands   ' + qc.list2string(channels, '%6s') + ' | ' + 'per band\n'
    result += '-' * hlen + '\n'
    result += 'delta   ' + qc.list2string(data_delta, '%6.2f') + ' | %6.2f\n' % np.sum(data_delta)
    result += 'theta   ' + qc.list2string(data_theta, '%6.2f') + ' | %6.2f\n' % np.sum(data_theta)
    result += 'mu      ' + qc.list2string(data_mu, '%6.2f') + ' | %6.2f\n' % np.sum(data_mu)
    #result += 'beta    ' + qc.list2string(data_beta, '%6.2f') + ' | %6.2f\n' % np.sum(data_beta)
    result += 'beta1   ' + qc.list2string(data_beta1, '%6.2f') + ' | %6.2f\n' % np.sum(data_beta1)
    result += 'beta2   ' + qc.list2string(data_beta2, '%6.2f') + ' | %6.2f\n' % np.sum(data_beta2)
    result += 'beta3   ' + qc.list2string(data_beta3, '%6.2f') + ' | %6.2f\n' % np.sum(data_beta3)
    result += 'lgamma  ' + qc.list2string(data_lgamma, '%6.2f') + ' | %6.2f\n' % np.sum(data_lgamma)
    result += 'hgamma  ' + qc.list2string(data_hgamma, '%6.2f') + ' | %6.2f\n' % np.sum(data_hgamma)
    result += '-' * hlen + '\n'
    result += 'per_ch  ' + qc.list2string(data_per_ch, '%6.2f') + ' | 100.00\n'
    print(result)
    p = qc.parse_path(featfile)
    open('%s/%s_summary.txt' % (p.dir, p.name), 'w').write(result)

    # export topo maps
    if topo_layout_file is not None:
        # default visualization setting
        res = 64
        contours = 6

        # select channel names to show
        if channel_name_show is None:
            channel_name_show = channels
        chan_vis = [''] * len(channels)
        for ch in channel_name_show:
            chan_vis[channels.index(ch)] = ch

        # set channel locations and reverse lookup table
        chanloc = {}
        if not os.path.exists(topo_layout_file):
            topo_layout_file = PYCNBI_ROOT + '/layout/' + topo_layout_file
            if not os.path.exists(topo_layout_file):
                raise FileNotFoundError('Layout file %s not found.' % topo_layout_file)
        logger.info('Using layout %s' % topo_layout_file)
        for l in open(topo_layout_file):
            token = l.strip().split('\t')
            ch = token[5]
            x = float(token[1])
            y = float(token[2])
            chanloc[ch] = [x, y]
        pos = np.zeros((len(channels),2))
        for i, ch in enumerate(channels):
            pos[i] = chanloc[ch]

        vmin = min(data_per_ch)
        vmax = max(data_per_ch)
        total = sum(data_per_ch)
        rate_delta = sum(data_delta) * 100.0 / total
        rate_theta = sum(data_theta) * 100.0 / total
        rate_mu = sum(data_mu) * 100.0 / total
        rate_beta = sum(data_beta) * 100.0 / total
        rate_beta1 = sum(data_beta1) * 100.0 / total
        rate_beta2 = sum(data_beta2) * 100.0 / total
        rate_beta3 = sum(data_beta3) * 100.0 / total
        rate_lgamma = sum(data_lgamma) * 100.0 / total
        rate_hgamma = sum(data_hgamma) * 100.0 / total
        export_topo(data_per_ch, pos, 'features_topo_all.png', xlabel='all bands 1-40 Hz', chan_vis=chan_vis)
        export_topo(data_delta, pos, 'features_topo_delta.png', xlabel='delta 1-3 Hz (%.1f%%)' % rate_delta, chan_vis=chan_vis)
        export_topo(data_theta, pos, 'features_topo_theta.png', xlabel='theta 4-7 Hz (%.1f%%)' % rate_theta, chan_vis=chan_vis)
        export_topo(data_mu, pos, 'features_topo_mu.png', xlabel='mu 8-12 Hz (%.1f%%)' % rate_mu, chan_vis=chan_vis)
        export_topo(data_beta, pos, 'features_topo_beta.png', xlabel='beta 13-30 Hz (%.1f%%)' % rate_beta, chan_vis=chan_vis)
        export_topo(data_beta1, pos, 'features_topo_beta1.png', xlabel='beta 12.5-16 Hz (%.1f%%)' % rate_beta1, chan_vis=chan_vis)
        export_topo(data_beta2, pos, 'features_topo_beta2.png', xlabel='beta 16-20 Hz (%.1f%%)' % rate_beta2, chan_vis=chan_vis)
        export_topo(data_beta3, pos, 'features_topo_beta3.png', xlabel='beta 20-28 Hz (%.1f%%)' % rate_beta3, chan_vis=chan_vis)
        export_topo(data_lgamma, pos, 'features_topo_lowgamma.png', xlabel='low gamma 31-40 Hz (%.1f%%)' % rate_lgamma, chan_vis=chan_vis)

def config_run(featfile=None, topo_layout_file=None):
    if featfile is None or len(featfile.strip()) == 0:
        if os.path.exists('good_features.txt'):
            featfile = os.path.realpath('good_features.txt').replace('\\', '/')
            logger.info('Found %s in the current folder.' % featfile)
        else:
            featfile = input('Feature file path? ')

    if topo_layout_file is None or len(topo_layout_file.strip()) == 0:
        topo_layout_file = 'antneuro_64ch.lay'

    feature_importances(featfile, topo_layout_file)

# sample code
if __name__ == '__main__':
    if len(sys.argv) > 2:
        config_run(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        config_run(sys.argv[1])
    else:
        config_run()
