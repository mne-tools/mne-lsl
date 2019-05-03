from __future__ import print_function, division

# 3: 'wrong', errp detected, -> positive
# 4: 'correct', no errp, -> negative

# %% General settings
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151028' # original 100
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151028/bkp' # 50 in the middle (bad)
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151118/fif'
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-Combined-ERRP_3_4' # combined two last dataset (220 trials)
# DATADIR= 'D:/Hoang/My Documents/Python/doudou-errp2/fif'
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151118/fif' # last dataset (120 trials)
# DATADIR= 'D:/Hoang/My Documents/Python/20151209-exg-glass-abs/fif'
# DATADIR= 'D:/Hoang/My Documents/Python/o8-20151209-absacc/fif/'
# DATADIR= 'D:/Hoang/My Documents/Python/sl-20151206-abs/fif/'
DATADIR = 'D:/Hoang/My Documents/Python/sl-20151213-absacc/fif/'
# DATADIR= 'D:/Hoang/My Documents/Python/201512-combined-absacc/'
# DATADIR= 'D:/Hoang/My Documents/Python/q5-20151209/absacc/fif/'

useLeaveOneOut = False
FILTER_METHOD = 'WINDOWED'  # LFILT (causal), WINDOWED (causal window based), NC (non-causal)
# True # Offline or online testing ? # todo: not active

APPLY_CAR = True
APPLY_PCA = True
# bbbAPPLY_OVERSAMPLING = False

OUTPUT_PCL = True

# USE_DIFF = False #use the differential of the signal as an additional feature
DO_CV = True  # perform cross-validation?

EXPORT_PLOTS = False  # todo remake this
noprint = False  # todo remake this

### Bandpass filter param
l_freq = 1.0
h_freq = 10.0

MAX_FPR = 0.2

### Features parameter
# picks_feat= [4,8,9,10,14] # kyuhwa initial
# picks_feat = [4,5,8,9,10,14] # iniaki,
# picks_feat = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # iniaki,
trig, Fz, FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4 = range(0, 17)
picks_feat = [1, 3, 4, 5, 8, 9, 10, 14]  # iniaki,
# picks_feat = [Fz,FC3,FC1,FCz,FC2,FC4,C3,C1,Cz,C2,C4,CP3,CP1,CPz,CP2,CP4]#everything
# picks_feat = [Fz,Cz,FCz,CPz]
# picks_feat = [Fz,    FC1,FCz,FC2,       C1,Cz,C2,           CPz]        #inaki normal
# picks_feat = [           FCz,    FC4,C3,C1,Cz,      C4,CP3,CP1,CPz,    CP4]
### Epoching parametersb
# offset = 0.3 # in seconds
offset = 0.0  # in seconds

tmin = 0.0 + offset  # tmin is redifinied afterward depending on baselining or not
tmin_bkp = tmin
# tmax= 0.8+offset # best for the 1st hundred
tmax = 0.7 + offset  # best for the 3rd hundred
paddingLength = 0.2  # padding time for bp
# tmax= 1+offset

# baselineRange = (-1+offset,0+offset)
# baselineRange = (-1,0) # None or (begin,end)
# baselineRange = (0,0.8)
baselineRange = None

decim_factor = 4  # take 1 data point every {decim_factor}
# regcoeff = 0.3 #best for 1st dataset
regcoeff = 0.7  # best for
# regcoeff = 0.3
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
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from pycnbi.triggers.trigger_def import trigger_def
from sklearn.metrics import roc_curve, auc, roc_auc_score
from plot_mne_epochs_grand_average import plot_grand_average
from scipy.signal import lfilter


# %%

def compute_features(signals, sfreq, l_freq, h_freq, decim_factor, shiftFactor, scaleFactor, pca, apply_online, tmin,
                     tmax, paddingIdx, iir_params):
    if apply_online is 'WINDOWED':
        signals_bp = mne.filter.band_pass_filter(signals, sfreq, l_freq, h_freq, method='fft', copy=True,
                                                 iir_params=None)
    else:
        signals_bp = signals
    signals_bp = signals_bp[:, :, paddingIdx:paddingIdx + (int((tmax - tmin) * sfreq))]  # crop the padding area for bp

    # Remove DC offset due to filtering
    for trial in range(signals_bp.shape[0]):
        for ch in range(signals_bp.shape[1]):
            signals_bp[trial, ch, :] = signals_bp[trial, ch, :] - np.mean(signals_bp[trial, ch, :])

    # Normalization
    signals_normalized = (signals_bp - shiftFactor) / scaleFactor

    # Downsample
    signals_downsampling = signals_normalized[:, :, ::decim_factor]

    # Merge channel and time dimension
    signals_reshaped = signals_downsampling.reshape(signals_downsampling.shape[0], -1)

    # PCA
    if pca is not None:
        signals_pcaed = pca.transform(signals_reshaped)
    else:
        signals_pcaed = signals_reshaped

    return signals_pcaed


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
    ### Utility parameter
    FLIST = qc.get_file_list(DATADIR, fullpath=True)
    n_jobs = mp.cpu_count()
    # n_jobs = 1 # for debug (if running in spyder)

    # %% Load data
    loadedraw, events = pu.load_multi(FLIST)
    raw = loadedraw.copy()
    processing_steps = []

    # Spatial filter - Common Average Reference (CAR)
    if APPLY_CAR:
        raw._data[1:] = raw._data[1:] - np.mean(raw._data[1:], axis=0)
        processing_steps.append('Car')

    tdef = trigger_def('triggerdef_errp.ini')
    sfreq = raw.info['sfreq']
    paddingIdx = int(paddingLength * sfreq)
    event_id = dict(correct=tdef.by_name['FEEDBACK_CORRECT'], wrong=tdef.by_name['FEEDBACK_WRONG'])

    # %% Dataset wide processing
    # Bandpass temporal filtering
    b, a, zi = pu.butter_bandpass(h_freq, l_freq, sfreq,
                                  raw._data.shape[0] - 1)  # raw._data.shape[0]- 1 because  channel 0 is trigger
    if FILTER_METHOD is 'NC':
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, picks=picks_feat, method='iir',
                   iir_params=None)  # method='iir'and irr_params=None -> filter with a 4th order Butterworth
    if FILTER_METHOD is 'LFILT':
        for x in range(1, raw._data.shape[0]):  # range starting from 1 because channel 0 is trigger
            # raw._data[x,:] = lfilter(b, a, raw._data[x,:])
            raw._data[x, :], zi[:, x - 1] = lfilter(b, a, raw._data[x, :], -1, zi[:, x - 1])
            # self.eeg[:,x], self.zi[:,x] = lfilter(b, a, self.eeg[:,x], -1,zi[:,x])

    # %% Epoching and baselining
    # epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baselineRange, picks=picks_feat, preload=True)
    t_lower = tmin - paddingLength
    t_upper = tmax + paddingLength
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=t_lower, tmax=t_upper, baseline=baselineRange,
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

    cv = StratifiedShuffleSplit(label, n_iter=20, test_size=0.1, random_state=1337)

    if useLeaveOneOut is True:
        cv = LeaveOneOut(len(label))

    if APPLY_PCA:
        processing_steps.append('PCA')
    # if APPLY_OVERSAMPLING:
    #	processing_steps.append('Oversampling')
    processing_steps.append('Normalization')
    processing_steps.append('Downsampling')


    # %% Fold processing
    def apply_cv(epochs):
        count = 1
        confusion_matrixes = []
        confusion_matrixes_percent = []
        predicted = ''
        test_label = ''
        firstIterCV = True
        probabilities = np.array([[]], ndmin=2)
        predictions = np.array([])
        best_threshold = []
        cv_probabilities = []
        cv_probabilities_label = []
        for train, test in cv:
            ## Train Data processing ##
            train_data = epochs._data[train]
            train_label = label[train]

            # Online simulation flag
            if FILTER_METHOD is 'WINDOWED':  # epochs should have one epoch only
                train_bp = mne.filter.band_pass_filter(train_data, sfreq, Fp1=2, Fp2=h_freq, copy=True,
                                                       filter_length=None, method='fft',
                                                       iir_params=None)  # bandpass on one epoch
            if FILTER_METHOD is 'NC' or FILTER_METHOD is 'LFILT':
                train_bp = train_data
            train_bp = train_bp[:, :, paddingIdx:paddingIdx + (int((tmax - tmin) * sfreq))]

            for trial in range(train_bp.shape[0]):
                for ch in range(train_bp.shape[1]):
                    train_bp[trial, ch, :] = train_bp[trial, ch, :] - np.mean(train_bp[trial, ch, :])

            # plt.figure()
            # plt.plot(train_bp[7,:].T)
            # plt.savefig(str(FILTER_METHOD)+'.png')
            # Normalization
            (train_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_bp, 'MinMax')

            # Downsampling
            train_downsampling = train_normalized[:, :, ::decim_factor]

            # Merge (reshape) channel and time for the PCA
            train_reshaped = train_downsampling.reshape(train_downsampling.shape[0], -1)

            # PCA initialisation
            if APPLY_PCA is False:
                pca = None
                train_pcaed = train_reshaped
            else:
                pca = PCA(0.95)
                pca.fit(train_reshaped)
                pca.components_ = -pca.components_  # inversion of vector to be constistant with Inaki's code
                train_pcaed = pca.transform(train_reshaped)

            # PCA
            #			train_pcaed = train_reshaped

            ## Test data processing ##
            test_data = epochs._data[test]
            test_label = label[test]

            # Compute_feature does the same steps as for train, but requires a computed PCA (that we got from train)
            # (bandpass, norm, ds, and merge channel and time)
            test_pcaed = compute_features(test_data, sfreq, l_freq, h_freq, decim_factor, trainShiftFactor,
                                          trainScaleFactor, pca, FILTER_METHOD, tmin, tmax, paddingIdx,
                                          iir_params=dict(a=a, b=b))
            #			test_pcaed = compute_features(test_data,sfreq,l_freq,h_freq,decim_factor,trainShiftFactor,trainScaleFactor,pca=None)

            ## Test ##
            train_x = train_pcaed
            test_x = test_pcaed

            # Classifier init
            #			RF = dict(trees=100, maxdepth=None)
            #			cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'], n_jobs=n_jobs)
            # cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'], class_weight="balanced", n_jobs=n_jobs)
            # cls = LDA(solver='eigen')
            #			cls = QDA(reg_param=0.3) # regularized LDA

            #			cls.fit( train_x, train_label )
            # Y_pred= cls.predict( test_x )
            # prediction = Y_pred

            # Fitting
            cls = rLDA(regcoeff)
            cls.fit(train_x, train_label)

            predicted = cls.predict(test_x)
            probs = cls.predict_proba(test_x)
            prediction = np.array(predicted)

            if useLeaveOneOut is True:
                if firstIterCV is True:
                    probabilities = np.append(probabilities, probs, axis=1)
                    firstIterCV = False
                    predictions = np.append(predictions, prediction)
                else:
                    probabilities = np.append(probabilities, probs, axis=0)
                    predictions = np.append(predictions, prediction)
            else:
                predictions = np.append(predictions, prediction)
                probabilities = np.append(probabilities, probs)

            # Performance
            if useLeaveOneOut is not True:
                cm = np.array(confusion_matrix(test_label, prediction))
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                confusion_matrixes.append(cm)
                confusion_matrixes_percent.append(cm_normalized)
                avg_confusion_matrixes = np.mean(confusion_matrixes_percent, axis=0)

            print('CV #' + str(count))
            print('Prediction: ' + str(prediction))
            print('    Actual: ' + str(test_label))

            # Append probs to the global list
            probs_np = np.array(probs)
            cv_probabilities.append(probs_np[:, 0])
            cv_probabilities_label.append(test_label)

            #			if useLeaveOneOut is not True:
            #				print('Confusion matrix')
            #				print(cm)
            #				print('Confusion matrix (normalized)')
            #				print(cm_normalized)
            #				print('---')
            #				print('True positive rate: '+str(cm_normalized[0][0]))
            #				print('True negative rate: '+str(cm_normalized[1][1]))
            print('===================')

            ## One CV done, go to the next one
            count += 1

        best_threshold = None
        cv_prob_linear = np.ravel(cv_probabilities)
        cv_prob_label_np = np.array(cv_probabilities_label)
        cv_prob_label_linear = np.ravel(cv_prob_label_np)
        threshold_list = np.linspace(0, 1, 100)

        biglist_fpr = []
        biglist_tpr = []
        biglist_thresh = []
        biglist_cms = []

        for thresh in threshold_list:
            biglist_pred = [4 if x < thresh else 3 for x in
                            cv_prob_linear]  # list comprehension to quickly go through the list.
            biglist_cm = confusion_matrix(cv_prob_label_linear, biglist_pred)
            biglist_cm_norm = biglist_cm.astype('float') / biglist_cm.sum(axis=1)[:, np.newaxis]
            biglist_cms.append(biglist_cm_norm)
            biglist_tpr.append(biglist_cm_norm[0][0])
            biglist_fpr.append(biglist_cm_norm[1][0])
            biglist_thresh.append(thresh)
        biglist_auc = auc(biglist_fpr, biglist_tpr)

        # Make a subset of data where FPR < MAX_FPR
        idx_below_maxfpr = np.where(np.array(biglist_fpr) < MAX_FPR)
        fpr_below_maxfpr = np.array(biglist_fpr)[idx_below_maxfpr[0]]
        tpr_below_maxfpr = np.array(biglist_tpr)[idx_below_maxfpr[0]]

        # Look for the best (max value) FPR in that subset
        best_tpr_below_maxfpr = np.max(tpr_below_maxfpr)
        best_tpr_below_maxfpr_idx = np.array(np.where(biglist_tpr == best_tpr_below_maxfpr)).ravel()  # get its idx

        # Get the associated TPRs
        best_tpr_below_maxfpr_associated_fpr = np.array(biglist_fpr)[best_tpr_below_maxfpr_idx]
        # Get the best (min value) in that subset
        best_associated_fpr = np.min(best_tpr_below_maxfpr_associated_fpr)
        # ... get its idx
        best_associated_fpr_idx = np.array(np.where(biglist_fpr == best_associated_fpr)).ravel()

        # The best idx is the one that is on both set
        best_idx = best_tpr_below_maxfpr_idx[np.in1d(best_tpr_below_maxfpr_idx, best_associated_fpr_idx)]

        plt.plot(biglist_fpr, biglist_tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        best_threshold = threshold_list[best_idx]
        print('#################################')
        print('Best treshold:' + str(best_threshold))
        print('Gives a TPR of ' + str(best_tpr_below_maxfpr))
        print('And a FPR of ' + str(best_associated_fpr))
        print('CM')
        print(biglist_cms[best_idx[0]])

        return (biglist_auc, best_threshold)


    if DO_CV:
        biglist_auc, best_threshold = apply_cv(epochs)

    if OUTPUT_PCL:
        ch_names = [raw.info['ch_names'][c] for c in picks_feat]

        train_data = epochs._data
        train_bp = mne.filter.band_pass_filter(train_data, sfreq, l_freq, h_freq, method='iir',
                                               iir_params=dict(a=a, b=b))  # bandpass
        train_bp = train_bp[:, :, paddingIdx:paddingIdx + (int((tmax - tmin) * sfreq))]
        (train_data_normalized, trainShiftFactor, trainScaleFactor) = normalizeAcrossEpoch(train_bp, 'MinMax')

        if True:
            train_data_downsampled = train_data_normalized[:, :, ::decim_factor]
            train_reshaped = train_data_downsampled.reshape(train_data_downsampled.shape[0],
                                                            -1)  # merge channel and time for the pca
            pca = PCA(0.95)
            pca.fit(train_reshaped)
            pca.components_ = -pca.components_
            train_pcaed = pca.transform(train_reshaped)

            # train classifier
            cls = rLDA(regcoeff)
            cls.fit(train_pcaed, label)

        if False:
            pca = None
            X = compute_features(train_data_normalized, sfreq, l_freq, h_freq, decim_factor, trainShiftFactor,
                                 trainScaleFactor, pca)
            # Classifier init
            RF = dict(trees=100, maxdepth=None)
            cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'],
                                         n_jobs=n_jobs)
            cls.fit(X, label)
            flen = X.shape[1]
            nch = len(picks_feat)
            step = int(flen / nch)
            assert flen % nch == 0
            ch_importance = []
            for i in range(0, flen, step):
                ch_importance.append(sum(cls.feature_importances_[i:i + step]))
            pass
        '''
        data= dict(sfreq=raw.info['sfreq'], ch_names=ch_names, picks=picks_feat, \
                w=w,b=b, l_freq=l_freq, h_freq=h_freq, decim_factor=decim_factor, pca=pca,
                shiftFactor=trainShiftFactor, scaleFactor=trainScaleFactor)
        '''
        # remember line 195:
        # t_lower = tmin-paddingLength
        # t_upper = tmax+paddingLength

        ##########################################################################
        data = dict(cls=cls, sfreq=raw.info['sfreq'], ch_names=ch_names, picks=picks_feat,\
                    l_freq=l_freq, h_freq=h_freq, decim_factor=decim_factor,\
                    shiftFactor=trainShiftFactor, scaleFactor=trainScaleFactor, pca=pca, threshold=best_threshold[0],
                    tmin=tmin, tmax=tmax, paddingIdx=paddingIdx, iir_params=dict(a=a, b=b))
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
