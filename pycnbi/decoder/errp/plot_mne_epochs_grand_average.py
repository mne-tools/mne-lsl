from __future__ import print_function, division

# -*- coding: utf-8 -*-
"""
Plot a grand average of a MNE Epoch that has a 'wrong'and a 'correct' event.
Also shows if the curve are statistically different for each time point.

Hoang Pham
"""

### Importation
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import matplotlib.patches as mpatches
import math
from scipy.stats import ttest_ind


def plot_grand_average(epochs, singleWindow=True, title='', alpha=0.05, correction='All', save=False):
    # %% Get the averages/means
    epavg = dict()
    for ev in epochs.event_id.keys():
        epavg[ev] = epochs[ev].average(None)

    # %% Add the difference
    epavg['difference'] = epavg['correct'] - epavg['wrong']  # in order to have an actual object
    epavg['difference'].data = epavg['correct'].data - epavg['wrong'].data
    # epavg['difference'].data = np.abs(epavg['correct'].data) - np.abs(epavg['wrong'].data)

    # %% ttest
    stat, pval = ttest_ind(epochs['wrong']._data, epochs['correct']._data, axis=0)

    # %% Mutiple test correction (Bonferroni and False Discovery Rate method)
    mne_reject_bonferroni = []
    mne_pval_bonferroni = []
    mne_reject_fdr_bh = []
    mne_pval_fdr_bh = []
    for channel in range(epochs._data.shape[1]):
        mne_reject_bonferroni_tmp, mne_pval_bonferonni_tmp = mne.stats.bonferroni_correction(pval[channel, :], alpha)
        mne_reject_fdr_bh_tmp, mne_pval_fdr_tmp = mne.stats.fdr_correction(pval[channel, :], alpha=alpha,
                                                                           method='indep')
        mne_reject_bonferroni.append(mne_reject_bonferroni_tmp)
        mne_pval_bonferroni.append(mne_pval_bonferonni_tmp)
        mne_reject_fdr_bh.append(mne_reject_fdr_bh_tmp)
        mne_pval_fdr_bh.append(mne_pval_fdr_tmp)
    mne_pval_bonferroni = np.array(mne_pval_bonferroni)
    mne_pval_fdr_bh = np.array(mne_pval_fdr_bh)

    # %% Get where is right and wrong different ?
    mne_idx_where_different_vanilla = []
    mne_idx_where_different_bonferroni = []
    mne_idx_where_different_fdr = []

    for channel in range(epochs._data.shape[1]):
        mne_idx_where_different_vanilla.append(np.where(pval[channel] < alpha)[0])
        mne_idx_where_different_bonferroni.append(np.where(mne_pval_bonferroni[channel] < alpha)[0])
        mne_idx_where_different_fdr.append(np.where(mne_pval_fdr_bh[channel] < alpha)[0])

    # %% Utility func
    def to_idx(mstime):
        # divide by 1000 (ms),
        return int((mstime * int(epochs.info['sfreq']) / 1000))

    # %% Plotting er se

    # Get a square window with all the figures

    # plt.xkcd() # uncomment to enable fun !

    if singleWindow is True:
        howmanyy = math.ceil(math.sqrt(len(epavg[epavg.keys()[0]].data)))
        howmanyx = math.ceil(math.sqrt(len(epavg[epavg.keys()[0]].data)))
        plt.figure()
    for chidx in range(len(epavg[epavg.keys()[0]].data)):

        if singleWindow is False:
            plt.figure()
            ax1 = plt.axes()
        else:
            ax1 = plt.subplot(howmanyx, howmanyy, chidx + 1)

        # Get channel name
        ch = epavg[epavg.keys()[0]].ch_names[chidx]
        plt.title(str(ch) + ' ' + title)

        # Plot the means
        ax1.plot(epavg['correct'].data[chidx, :], label='correct', color='green')
        ax1.plot(epavg['wrong'].data[chidx, :], label='wrong', color='red')
        ax1.plot(epavg['difference'].data[chidx, :], label='difference', color='black', linestyle=':')

        # Set the decorations
        plt.xticks(
            [to_idx(0), to_idx(50), to_idx(100), to_idx(200), to_idx(300), to_idx(500), to_idx(800), to_idx(1000)],
            ['0', '50', '100', '200', '300', '500', '800', '1000'], rotation=-90)
        plt.xlabel('Time [ms]')
        plt.ylabel('Potential [uV]')

        # Get the channel where there's the most difference
        chan_len_of_idx = []
        for chan in mne_idx_where_different_bonferroni:
            chan_len_of_idx.append(len(chan))

        # Plot where's it's significantly diifferent
        for idx in mne_idx_where_different_vanilla[chidx]:
            ax1.axvline(x=idx, linewidth=1, color='gray', alpha=0.2)
        if correction.lower() == 'bonferroni':
            for idx in mne_idx_where_different_bonferroni[chidx]:
                ax1.axvline(x=idx, linewidth=1, color='brown', alpha=0.2)
        if correction.lower() == 'fdr':
            for idx in mne_idx_where_different_fdr[chidx]:
                ax1.axvline(x=idx, linewidth=1, color='darkblue', alpha=0.2)
        if correction.lower() == 'all':
            for idx in mne_idx_where_different_bonferroni[chidx]:
                ax1.axvline(x=idx, linewidth=1, color='brown', alpha=0.2)
            for idx in mne_idx_where_different_fdr[chidx]:
                ax1.axvline(x=idx, linewidth=1, color='darkblue', alpha=0.2)
        # fisher score from 0.2 to 0.8
        fisher = []
        #        showPVal=True
        #        if showPVal is True:
        #            print('hey im in')
        #            ax2 = ax1.twinx()
        #            ax2.plot(mne_pval_fdr_bh[chidx],alpha=0.9,color='darkred')
        #            ax2.hold(True)
        #            ax2.plot(mne_pval_bonferroni[chidx],alpha=0.9, color='teal')
        #            ax2.plot(pval[chidx],alpha=0.9, color='pink')
        #            ax2.hold(False)
        #	        plt.ylabel('P-Value')

        # Set Legend
        correctLineLegend = mlines.Line2D([], [], color='green', label='Correct')
        wrongLineLegend = mlines.Line2D([], [], color='red', label='Wrong')
        differenceLineLegend = mlines.Line2D([], [], color='black', linestyle=':', label='Difference')
        statDiffHighlight = mpatches.Patch(color='gray', alpha=0.2, label='p<0.05')
        legendsList = [correctLineLegend, wrongLineLegend, differenceLineLegend, statDiffHighlight]
        if correction.lower() == 'bonferroni':
            statDiffHighlight_bonferroni = mpatches.Patch(color='brown', alpha=0.2, label='p<0.05 (Bonf. corr)')
            legendsList.append(statDiffHighlight_bonferroni)
        if correction.lower() == 'fdr':
            statDiffHighlight_fdr = mpatches.Patch(color='darkblue', alpha=0.2, label='p<0.05 (FDR corr)')
            legendsList.append(statDiffHighlight_fdr)

        # Show legend (multi window)
        if singleWindow is False:
            plt.legend(handles=legendsList, loc=2, prop={'size':10})
        if save is True:
            plt.savefig(str(ch))

    # Set ticks and axis label
    plt.tight_layout()

    # Show legend (single window)
    if singleWindow is True:
        plt.legend(handles=legendsList, bbox_to_anchor=(2, 1))


# Debug
if __name__ == '__main__':
    #	DATADIR= 'D:/data/ErrP/q5/20151224/1-passive-200/fif/'
    DATADIR = 'D:/data/ErrP/q5/20160114/100-train/fif/'
    l_freq = 1.0
    h_freq = 10.0
    import pycnbi
    import pycnbi.utils.pycnbi_utils as pu
    import pycnbi.utils.q_common as qc
    import multiprocessing as mp

    FLIST = qc.get_file_list(DATADIR, fullpath=True)
    n_jobs = mp.cpu_count()
    raw, events = pu.load_multi(FLIST)
    trig, Fz, FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4 = range(0, 17)
    picks_feat = [Fz, FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4]
    picks_feat = [1, 3, 4, 5, 8, 9, 10, 14]  # iniaki,
    # picks_feat = [1]
    from pycnbi.triggers.trigger_def import trigger_def

    tdef = trigger_def('triggerdef_errp.ini')
    sfreq = raw.info['sfreq']
    # baselineRange = None
    baselineRange = None

    tmin = 0.0
    tmax = 1.0

    ONLINE = False

    event_id = dict(correct=tdef.by_name['FEEDBACK_CORRECT'], wrong=tdef.by_name['FEEDBACK_WRONG'])
    if ONLINE is False:
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, picks=picks_feat, method='iir',
                   iir_params=None)  # method='iir'and irr_params=None -> filter with a 4th order Butterworth

    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baselineRange,
                        picks=picks_feat, preload=True, proj=False)

    epochs_online = epochs.copy()
    for epoch in epochs_online._data:
        mne.filter.band_pass_filter(epoch, sfreq, l_freq, h_freq, method='fft', copy=False)
    if ONLINE is True:
        plot_grand_average(epochs_online, correction='bonferroni')
    else:
        plot_grand_average(epochs, singleWindow=True, correction='bonferroni', save=False, title='abs')

        ############################################################

        #	DATADIR= 'D:/Hoang/My Documents/Python/sl-20151213-absacc/fif/'
        #	l_freq= 1.0
        #	h_freq= 10.0
        #	import pycnbi
        #	import pycnbi.utils.pycnbi_utils as pu
        #	import pycnbi.utils.q_common as qc
        #	import multiprocessing as mp
        #	FLIST= qc.get_file_list(DATADIR)
        #	n_jobs= mp.cpu_count()
        #	raw, events= pu.load_multi(FLIST)
        #	trig,Fz,FC3,FC1,FCz,FC2,FC4,C3,C1,Cz,C2,C4,CP3,CP1,CPz,CP2,CP4=range(0,17)
        #	picks_feat = [Fz,FC3,FC1,FCz,FC2,FC4,C3,C1,Cz,C2,C4,CP3,CP1,CPz,CP2,CP4]
        #	picks_feat = [1,3,4,5,8,9,10,14] # iniaki,
        #	from pycnbi.triggers.trigger_def import trigger_def
        #	tdef = trigger_def('triggerdef_errp.ini')
        #	sfreq= raw.info['sfreq']
        #	#baselineRange = None
        #	baselineRange = None
        #
        #	tmin= 0.0
        #	tmax= 1.0
        #
        #	ONLINE = False
        #
        #	event_id= dict(correct=tdef.by_name['FEEDBACK_CORRECT'], wrong=tdef.by_name['FEEDBACK_WRONG'])
        #	if ONLINE is False:
        #		raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, picks=picks_feat, method='iir', iir_params=None) # method='iir'and irr_params=None -> filter with a 4th order Butterworth
        #
        #	epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,baseline=baselineRange, picks=picks_feat, preload=True, proj=False)
        #
        #	epochs_online = epochs.copy()
        #	for epoch in epochs_online._data:
        #		mne.filter.band_pass_filter(epoch, sfreq, l_freq, h_freq, method='fft',copy=False)
        #	if ONLINE is True:
        #		plot_grand_average(epochs_online,correction='bonferroni')
        #	else:
        #		plot_grand_average(epochs,singleWindow=True,correction='bonferroni',save=False,title='absacc')
        #
        #	#plotGrandAverage(epochs,singleWindow=False,title='Foo')
        #	#import seaborn as sns
        #	#plt.figure();sns.tsplot(data=epochs['correct']._data[:,0,:],ci=[100,95,90,85,80,75,70,65,60,55,50])
