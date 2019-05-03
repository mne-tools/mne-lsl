from __future__ import print_function, division

"""
TODO:
	normalize
	baseline for online
	match the number of samples for final training
	window-based filtering -> online test
"""

# general settings
DATADIR = 'D:/data/ErrP/hoang/20151005/fif'
APPLY_BASELINE = False  # subtract mean from epochs for better visualization
FILTER_ONLINE = True  # True: apply spectral filter after epoching. False: apply filter on entire signal
DO_CV = True  # perform cross-validation?
EXPORT_PLOTS = False
CLASSIFIER = 'RLDA'  # RLDA, RF

# bandpass filter
l_freq = 1.0
h_freq = 10.0

# features
sfreq_feat = 64  # downsample to this Hz
# picks_feat= [1,3,4,5,8,9,10,14]
picks_feat = [4, 8, 9, 10, 14]

import pycnbi
import os, sys, random
import pycnbi.utils.q_common as qc
import numpy as np
import mne
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from pycnbi.decoder.rlda import rLDA_binary
from pycnbi.triggers.trigger_def import trigger_def

FLIST = qc.get_file_list(DATADIR, fullpath=True)
n_jobs = mp.cpu_count()


# get grand averages of epochs
def plot_grand_avg(epochs, outdir, picks=None):
    qc.make_dirs(outdir)
    for ev in epochs.event_id.keys():
        epavg = epochs[ev].average(picks)
        for chidx in range(len(epavg.ch_names)):
            ch = epavg.ch_names[chidx]
            epavg.plot(picks=[chidx], unit=True, ylim=dict(eeg=[-10, 10]), titles='%s-%s' % (ch, ev),\
                       show=False, scalings={'eeg':1}).savefig(outdir + '/%s-%s.png' % (ch, ev))


# compute features for the classifier
def get_features(epochs_in, picks, decim_factor):
    # downsample
    epochs = epochs_in.decimate(decim_factor, copy=True)

    feat = {}
    for ev in epochs.event_id.keys():
        epdata = epochs[ev].get_data()[:, picks, :]
        feat[ev] = epdata.reshape(epdata.shape[0], -1)
    return feat


# filter, downsample and compute features
# signals: trials x channels x samples
def get_features_window(signals, sfreq, l_freq, h_freq, decim_factor):
    # bandpass filter
    signals_bp = mne.filter.band_pass_filter(signals, sfreq, l_freq, h_freq, method='iir')

    # downsample
    n_tr = signals.shape[0]
    n_ch = signals.shape[1]
    n_sp = signals.shape[2]

    # trim to the closest divisible length
    mod = n_sp % decim_factor
    if not mod == 0:
        signals_bp = signals_bp[:, :, :-mod]
    signals_ds = signals_bp.reshape(n_tr, n_ch, -1, decim_factor)[:, :, :, 0]

    # concatenate
    feat = signals_ds.reshape(n_tr, -1)

    return feat


def balance_idx(label):
    labelsetWrong = np.where(label == 3)[0]  ##############################
    labelsetCorrect = np.where(label == 4)[0]  ##############################

    diff = len(labelsetCorrect) - len(labelsetWrong)

    if diff > 0:
        smallestSet = labelsetWrong
        largestSet = labelsetCorrect
    elif diff < 0:
        smallestSet = labelsetCorrect
        largestSet = labelsetWrong

    idx_for_balancing = []

    for i in range(diff):
        idx_for_balancing.append(random.choice(smallestSet))

    return idx_for_balancing


def normalizeAcrossEpoch(epoch_data, method, givenShiftFactor=0, givenScaleFactor=1):
    # Normalize across epoch
    # Assumes epoch_data have form ({trial}L, [channel]L,{time}L)

    new_epochs_data = epoch_data
    shiftFactor = 0
    scaleFactor = 0

    # This way of doing obviously only work if you shift and scale your data (is there other ways ?)
    if method == 'zScore':
        shiftFactor = np.mean(epoch_data, 0)
        scaleFactor = np.std(epoch_data, 0)
    elif method == 'MinMax':
        shiftFactor = np.max(epoch_data, 0)
        scaleFactor = np.max(epoch_data, 0) - np.min(epoch_data, 0)
    elif method == 'override':  # todo: find a better name
        shiftFactor = givenShiftFactor
        scaleFactor = givenScaleFactor

    for trial in range(new_epochs_data.shape[0]):
        new_epochs_data[trial, :, :] = (new_epochs_data[trial, :, :] - shiftFactor) / scaleFactor

    return (new_epochs_data, shiftFactor, scaleFactor)


if __name__ == '__main__':
    # load data
    raw, events = pu.load_multi(FLIST, spfilter='car')
    tdef = trigger_def('triggerdef_errp.ini')
    sfreq = raw.info['sfreq']

    # epoching
    tmin = 0
    tmax = 1
    event_id = dict(correct=tdef.by_name['FEEDBACK_CORRECT'], wrong=tdef.by_name['FEEDBACK_WRONG'])

    # export plots: apply offline spectral filter
    if EXPORT_PLOTS == True:
        # apply filter on entire signal (for offline analysis)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, picks=picks_feat, method='iir', iir_params=None)

        # apply baseline
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(None, 0),
                            picks=picks_feat, preload=True)

        # plot
        print('Plotting grand averages.')
        plot_grand_avg(epochs, DATADIR + '/figs')

        print('Quitting after exporting plots.')
        sys.exit()

    # do not apply baseline
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, picks=picks_feat,
                        preload=True)

    # compute features
    decim_factor = int(sfreq / sfreq_feat)
    if FILTER_ONLINE == True:
        # filter on each window (simulate online)
        feats = {}
        feats['correct'] = get_features_window(epochs['correct'].get_data(), sfreq, l_freq, h_freq, decim_factor)
        feats['wrong'] = get_features_window(epochs['wrong'].get_data(), sfreq, l_freq, h_freq, decim_factor)
    else:
        # filter on entire signals (offline)
        feats = get_features(epochs, None, decim_factor)
    X = np.concatenate((feats['correct'], feats['wrong']), axis=0)
    Y = np.array([event_id['correct']] * feats['correct'].shape[0] + [event_id['wrong']] * feats['wrong'].shape[0])

    # cross-validation
    if DO_CV == True:
        # cv= StratifiedShuffleSplit(Y, n_iter=20, test_size=0.2)


        # hoang's code
        label = epochs.events[:, 2]
        cv = StratifiedShuffleSplit(label, n_iter=20, test_size=0.2)

        count = 0
        scores = []
        cm_norm_avg = None
        for train, test in cv:
            count += 1

            if CLASSIFIER == 'RF':
                # random forest
                RF = dict(trees=1000, maxdepth=100)
                cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'],
                                             n_jobs=n_jobs)
            elif CLASSIFIER == 'RLDA':
                cls = rLDA_binary(0.3)

            train_data = epochs._data[train]
            train_label = label[train]
            test_data = epochs._data[test]
            test_label = label[test]
            # train_data= X[train]
            # test_data= X[test]
            # train_label= Y[train]
            # test_label= Y[test]

            # from Hoang's code
            ### Normalization
            (train_data_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_data, 'MinMax')
            (test_data_normalized, testShiftFactor, testScaleFactor) = normalizeAcrossEpoch(test_data, 'override',
                                                                                            trainShiftFactor,
                                                                                            trainScaleFactor)

            ### Downsampling
            train_data_downsampled = train_data_normalized[:, :, ::decim_factor]
            test_data_downsampled = test_data_normalized[:, :, ::decim_factor]

            train_x = train_data_downsampled.reshape(train_data_downsampled.shape[0],
                                                     -1)  # put the last dimension into the preceding one
            test_x = test_data_downsampled.reshape(test_data_downsampled.shape[0],
                                                   -1)  # put the last dimension into the preceding one

            # next: apply PCA
            if True:
                pca = PCA(0.95)
                pca.fit(train_x)
                pca.components_ = -pca.components_  #
                train_x = pca.transform(train_x)
                test_x = pca.transform(test_x)

            # oversampling the least present sample
            if False:
                idx_offset = balance_idx(train_label)
                oversampled_train_label = np.append(train_label, train_label[idx_offset])
                oversampled_train_x = np.concatenate((train_x, train_x[idx_offset]), 0)
                train_label = oversampled_train_label
                train_x = oversampled_train_x

            cls.fit(train_x, np.unique(train_label))
            # cls.fit( oversampled_train_x, oversampled_train_label )

            Y_pred = cls.predict(test_x)
            cm = confusion_matrix(Y[test], Y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            if cm_norm_avg is None:
                cm_norm_avg = cm_normalized
            else:
                cm_norm_avg += cm_normalized

            print(cm_normalized)
        # scores.append( cls.score( X[test], Y[test] ) )
        # print('Cross-validation %d: %.2f'% (count,scores[-1]) )
        print('\nAveraged over %d folds' % count)
        print(cm_norm_avg / count)
    # print('Mean accuracy: %.2f'% np.mean(scores) )

    # train using entire dataset
    if False:
        # train a model
        RF = dict(trees=1000, maxdepth=100)
        cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'],
                                     n_jobs=n_jobs)
        cls.fit(X, Y)
        cls.n_jobs = 1  # n_jobs should be 1 for online decoding
        print('Trained a Random Forest classifer with %d trees and %d maxdepth' % (RF['trees'], RF['maxdepth']))
        ch_names = [raw.info['ch_names'][c] for c in picks_feat]
        data = dict(sfreq=raw.info['sfreq'], ch_names=ch_names, picks=picks_feat,\
                    cls=cls, l_freq=l_freq, h_freq=h_freq, decim_factor=decim_factor)
        outdir = DATADIR + '/errp_classifier'
        qc.make_dirs(outdir)
        clsfile = outdir + '/errp_classifier.pcl'
        qc.save_obj(clsfile, data)
        print('Saved as %s' % clsfile)

    if True:
        # hoang's code
        label = epochs.events[:, 2]

        cls = rLDA_binary(0.3)

        train_data = epochs._data
        train_label = label

        ### Normalization
        (train_data_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_data, 'MinMax')

        ### Downsampling
        train_data_downsampled = train_data_normalized[:, :, ::decim_factor]

        train_x = train_data_downsampled.reshape(train_data_downsampled.shape[0],
                                                 -1)  # put the last dimension into the preceding one

        # next: apply PCA
        if True:
            pca = PCA(0.95)
            pca.fit(train_x)
            pca.components_ = -pca.components_  #
            train_x = pca.transform(train_x)

        # oversampling the least present sample
        if False:
            idx_offset = balance_idx(train_label)
            oversampled_train_label = np.append(train_label, train_label[idx_offset])
            oversampled_train_x = np.concatenate((train_x, train_x[idx_offset]), 0)
            train_label = oversampled_train_label

        cls.fit(train_x, np.unique(train_label))

        # complete the info later
        errpdata = dict(trainShiftFactor=trainShiftFactor, trainScaleFactor=trainScaleFactor, pca=pca)
        cls_dir = DATADIR + '/classifier/'
        qc.make_dirs(cls_dir)
        qc.save_obj(cls_dir + 'classifier_rlda.pcl', dict(cls=cls))
        qc.save_obj(cls_dir + 'hoang_errp_params.pcl', errpdata)

    print('Done.')
