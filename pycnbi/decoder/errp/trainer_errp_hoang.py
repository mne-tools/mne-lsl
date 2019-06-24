from __future__ import print_function, division

# %% General settings
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151028'
DATADIR = 'D:/data/ErrP/hoang/20151005/fif'

SIMULATE_ONLINE = True  # Offline or online testing ? # todo: not active

APPLY_CAR = False
APPLY_PCA = True
APPLY_OVERSAMPLING = False

OUTPUT_PCL = True

# USE_DIFF = False #use the differential of the signal as an additional feature

DO_CV = True  # perform cross-validation?

EXPORT_PLOTS = False  # todo remake this
noprint = False  # todo remake this

### Bandpass filter param
l_freq = 1.0
h_freq = 10.0

### Features parameter
# picks_feat= [4,8,9,10,14] # kyuhwa initial
picks_feat = [1, 3, 4, 5, 8, 9, 10, 14]  # iniaki, normal
# picks_feat = list(range(1,17))

### Epoching parametersb
# offset = 0.3 # in seconds
offset = 0.0  # in seconds

tmin = 0 + offset  # tmin is redifinied afterward depending on baselining or not
tmin_bkp = tmin
tmax = 0.8 + offset

# baselineRange = (-1+offset,0+offset)
# baselineRange = (-1,0) # None or (begin,end)
# baselineRange = (0,0.8)
baselineRange = None

decim_factor = 4  # take 1 data point every {decim_factor}

# Baseline correction. Assume that baseline[0] is before baseline[1]
if baselineRange and baselineRange[0] < tmin:
    tmin = baselineRange[0]
normRange = (tmin, 1 + offset)

### Importation
import os, sys
import numpy as np
import random
import mne
import matplotlib.pyplot as plt
import multiprocessing as mp

import pycnbi
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
from pycnbi.decoder.rlda import rLDA

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from pycnbi.triggers.trigger_def import trigger_def

### Utility parameter
FLIST = qc.get_file_list(DATADIR, fullpath=True)
n_jobs = mp.cpu_count()


# n_jobs = 1 # for debug (if running in spyder)

# %%
def trainLDA(X, Y, reg_cov=None):
    indexError = np.where(Y == 3)[0]
    indexCorrect = np.where(Y == 4)[0]
    cov = np.matrix(np.cov(X.T))
    mu1 = np.matrix(np.mean(X[indexError], axis=0).T).T
    mu2 = np.matrix(np.mean(X[indexCorrect], axis=0).T).T
    mu = (mu1 + mu2) / 2;
    numFeatures = X.shape[1]

    if reg_cov is not None and numFeatures > 1:
        if reg_cov < 1:
            lambdaStar = reg_cov
        else:
            assert False

        cov = (1 - lambdaStar) * cov + (lambdaStar / numFeatures) * np.trace(cov) * np.eye(cov.shape[0])

    w = np.linalg.pinv(cov) * (mu2 - mu1)
    b = -(w.T) * mu

    return w, b


def compute_features(signals, sfreq, l_freq, h_freq, decim_factor, shiftFactor, scaleFactor, pca):
    if SIMULATE_ONLINE is True:
        signals_bp = mne.filter.band_pass_filter(signals, sfreq, l_freq, h_freq, method='iir')
    else:
        signals_bp = signals

    # Normalization
    print(signals_bp.shape)
    print('----')
    print(signals.shape)
    print('----')
    print(shiftFactor.shape)
    print('----')
    print(scaleFactor.shape)
    signals_normalized = (signals_bp - shiftFactor) / scaleFactor

    # Downsample
    signals_downsampling = signals_normalized[:, :, ::decim_factor]

    # Merge channel and time dimension
    signals_reshaped = signals_downsampling.reshape(signals_downsampling.shape[0], -1)

    # PCA
    signals_pcaed = pca.transform(signals_reshaped)

    return signals_pcaed


def testLDA(X, w, b):
    probs = []
    predicted = []
    for row in X:
        probability = w.T * np.matrix(row).T + b.T
        probs.append(probability)
        if probability >= 0:  # correct
            predicted.append(4)
        else:  # wrong
            predicted.append(3)

    return predicted, probs


# %%

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

    if len(new_epochs_data.shape) == 3:
        for trial in range(new_epochs_data.shape[0]):
            new_epochs_data[trial, :, :] = (new_epochs_data[trial, :, :] - shiftFactor) / scaleFactor
    else:
        new_epochs_data = (new_epochs_data - shiftFactor) / scaleFactor

    return (new_epochs_data, shiftFactor, scaleFactor)


if __name__ == '__main__':

    # %% Load data
    raw, events = pu.load_multi(FLIST)
    processing_steps = []

    if APPLY_CAR:
        raw._data[1:] = raw._data[1:] - np.mean(raw._data[1:], axis=0)
        processing_steps.append('Car')

    tdef = trigger_def('triggerdef_errp.ini')
    sfreq = raw.info['sfreq']
    event_id = dict(correct=tdef.by_name['FEEDBACK_CORRECT'], wrong=tdef.by_name['FEEDBACK_WRONG'])
    # %% simu online
    SIMULATE_ONLINE = False

    #    if SIMULATE_ONLINE is True:
    #        signal = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,baseline=baselineRange, picks=picks_feat, preload=True, proj=False)
    #        signal_data = np.reshape(signal[0]._data,(signal[0]._data.shape[1],signal[0]._data.shape[2]))
    #        foo = compute_features(signal_data,sfreq,l_freq,h_freq,decim_factor)
    #        print('finalfeat')
    #        print(foo)

    # Spatial filter - Common Average Reference (CAR) # todo: reactivate

    # %% Dataset wide processing
    # Bandpass temporal filtering
    if SIMULATE_ONLINE is False:
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, picks=picks_feat, method='iir',
                   iir_params=None)  # method='iir'and irr_params=None -> filter with a 4th order Butterworth

    # %% Epoching and baselining
    # epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baselineRange, picks=picks_feat, preload=True)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baselineRange,
                        picks=picks_feat, preload=True, proj=False)

    if (baselineRange):
        processing_steps.append('Baselining')
    if tmin != tmin_bkp:
        # if the baseline range was before the initial tmin, epochs was tricked to
        # to select this range (it expects that baseline is witin [tmin,tmax])
        # this part restore the initial tmin and prune the data
        epochs.tmin = tmin_bkp
        epochs._data = epochs._data[:, :, int((tmin_bkp - tmin) * sfreq):]

    # %% Fold creation
    # epochs.events contains the label that we want on the third column
    # We can then get the relevent data within a fold by doing epochs._data[test]
    # It will return an array with size ({test}L, [channel]L,{time}L)
    label = epochs.events[:, 2]

    cv = StratifiedShuffleSplit(label, n_iter=20, test_size=0.2, random_state=0)
    count = 1
    scores = []
    confusion_matrixes = []
    confusion_matrixes_percent = []
    tn_rates = []
    tp_rates = []
    predicted = ''
    test_label = ''

    if APPLY_PCA:
        processing_steps.append('PCA')
    if APPLY_OVERSAMPLING:
        processing_steps.append('Oversampling')
    processing_steps.append('Normalization')
    processing_steps.append('Downsampling')

    # %% Fold processing
    if DO_CV:
        for train, test in cv:

            ## Train & Parameter computation
            train_data = epochs._data[train]
            train_label = label[train]

            if SIMULATE_ONLINE is True:
                train_bp = mne.filter.band_pass_filter(train_data, sfreq, l_freq, h_freq, method='iir')  # bandpass
            else:
                train_bp = train_data
            (train_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_bp,
                                                                                          'MinMax')  # normalization

            train_downsampling = train_normalized[:, :, ::decim_factor]  # downsampling

            train_reshaped = train_downsampling.reshape(train_downsampling.shape[0],
                                                        -1)  # merge channel and time for the pca
            pca = PCA(0.95)
            pca.fit(train_reshaped)
            pca.components_ = -pca.components_

            train_pcaed = pca.transform(train_reshaped)  # pca
            train_x = train_pcaed
            ## Test
            test_data = epochs._data[test]
            test_label = label[test]

            # compute_feature does bandpass, norm, ds and merge channel and time
            test_pcaed = compute_features(test_data, sfreq, l_freq, h_freq, decim_factor, trainShiftFactor,
                                          trainScaleFactor, pca)

            test_x = test_pcaed

            # oversampling the least present sample
            if APPLY_OVERSAMPLING:
                idx_offset = balance_idx(train_label)
                oversampled_train_label = np.append(train_label, train_label[idx_offset])
                oversampled_train_x = np.concatenate((train_x, train_x[idx_offset]), 0)
                train_label = oversampled_train_label
                train_x = oversampled_train_x

            # RF = dict(trees=1000, maxdepth=None)
            # cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'], n_jobs=n_jobs)
            # cls = LDA(solver='eigen')
            # cls = QDA(reg_param=0.3) # regularized LDA


            # w,b = trainLDA(train_x,train_label, 0.3)
            # predicted, probs = testLDA(test_x, w, b)

            rlda = rLDA()
            rlda.fit(train_x, train_label, 0.3)
            predicted, probs = rlda.predict_proba(test_x)

            prediction = np.array(predicted)

            #            cls.fit( train_x, train_label )
            #            Y_pred= cls.predict( test_x )
            #            prediction = Y_pred

            cm = np.array(confusion_matrix(test_label, prediction))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            tp_rates.append(cm_normalized[0][0])
            tn_rates.append(cm_normalized[1][1])
            confusion_matrixes.append(cm)
            confusion_matrixes_percent.append(cm_normalized)

            print('CV #' + str(count))
            print('Prediction: ' + str(prediction))
            print('    Actual: ' + str(test_label))
            print('Confusion matrix')
            print(cm)
            print('Confusion matrix (normalized)')
            print(cm_normalized)
            print('---')
            print('True positive rate: ' + str(cm_normalized[0][0]))
            print('True negative rate: ' + str(cm_normalized[1][1]))
            print('===================')
            count += 1
        print('Summary')
        print('-------')
        print('Processing steps: ' + str(processing_steps))
        print('Average Confusion matrix')
        print(np.mean(confusion_matrixes_percent, axis=0))
        print('Average true positive rate: ' + str(np.mean(tp_rates)))  # prediction of "wrong" as "wrong"
        print('Average true negative rate: ' + str(np.mean(tn_rates)))  # prediction of "correct" as "correct"

    if OUTPUT_PCL:
        ch_names = [raw.info['ch_names'][c] for c in picks_feat]

        train_data = epochs._data
        train_bp = mne.filter.band_pass_filter(train_data, sfreq, l_freq, h_freq, method='iir')  # bandpass
        (train_data_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_bp, 'MinMax')
        train_data_downsampled = train_data_normalized[:, :, ::decim_factor]
        train_reshaped = train_data_downsampled.reshape(train_data_downsampled.shape[0],
                                                        -1)  # merge channel and time for the pca
        pca = PCA(0.95)
        pca.fit(train_reshaped)
        pca.components_ = -pca.components_
        train_pcaed = pca.transform(train_reshaped)
        w, b = trainLDA(train_pcaed, label, 0.3)

        data = dict(sfreq=raw.info['sfreq'], ch_names=ch_names, picks=picks_feat,\
                    w=w, b=b, l_freq=l_freq, h_freq=h_freq, decim_factor=decim_factor, pca=pca,
                    shiftFactor=trainShiftFactor, scaleFactor=trainScaleFactor)
        outdir = DATADIR + '/errp_classifier'
        qc.make_dirs(outdir)
        clsfile = outdir + '/errp_classifier.pcl'
        qc.save_obj(clsfile, data)
        print('Saved as %s' % clsfile)
print('Done')






#    def balance_idx(label):
#        labelsetWrong = np.where(label==3)[0]
#        labelsetCorrect = np.where(label==4)[0]
#
#        diff = len(labelsetCorrect) - len(labelsetWrong)
#
#        if diff > 0:
#            smallestSet = labelsetWrong
#            largestSet = labelsetCorrect
#        elif diff<0:
#            smallestSet = labelsetCorrect
#            largestSet = labelsetWrong
#
#        idx_for_balancing = []
#
#        for i in range(diff):
#            idx_for_balancing.append(random.choice(smallestSet))
#
#        return idx_for_balancing
#
