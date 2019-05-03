from __future__ import print_function, division

# 3: 'wrong', errp detected, -> positive
# 4: 'correct', no errp, -> negative

# %% General settings
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151028' # original 100
# DATADIR= 'D:/Hoang/My Documents/Python/Hoang-ErrP20151117/fif'
DATADIR = 'D:/Hoang/My Documents/Python/Hoang-Combined-ERRP_3_4'

SIMULATE_ONLINE = False  # Offline or online testing ? # todo: not active

APPLY_CAR = False
APPLY_PCA = True
APPLY_OVERSAMPLING = False

OUTPUT_PCL = True

# USE_DIFF = False #use the differential of the signal as an additional feature

DO_CV = True  # perform cross-validation?

EXPORT_PLOTS = False  # todo remake this
noprint = False  # todo remake this
MAX_FPR = 0.1
### Bandpass filter param
l_freq = 1.0
h_freq = 10.0

### Features parameter
# picks_feat= [4,8,9,10,14] # kyuhwa initial
trig, Fz, FC3, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4 = range(0, 16)
picks_feat = [1, 3, 4, 5, 8, 9, 10, 14]  # iniaki, normal
picks_feat = [Fz, FC3, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4]
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

useLeaveOneOut = False

decim_factor = 8  # take 1 data point every {decim_factor}

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

# Silence mne
mne.set_log_level(False)

### Utility parameter
FLIST = qc.get_file_list(DATADIR, fullpath=True)
n_jobs = mp.cpu_count()


# n_jobs = 1 # for debug (if running in spyder)

# %%

def compute_features(signals, sfreq, l_freq, h_freq, decim_factor, shiftFactor, scaleFactor, pca):
    if SIMULATE_ONLINE is True:
        signals_bp = mne.filter.band_pass_filter(signals, sfreq, l_freq, h_freq, method='iir')
    else:
        signals_bp = signals

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

    # %% Load data
    loadedraw, events = pu.load_multi(FLIST)
    processing_steps = []


    def getperf(APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV, l_freq, h_freq, picks_feat, offset, tmin, tmax,
                baselineRange, reg_coeff, verbose):
        raw = loadedraw.copy()
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
                            picks=picks_feat, preload=True, proj=False, verbose=False)

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

        cv = StratifiedShuffleSplit(label, n_iter=20, test_size=0.2, random_state=1337)

        if useLeaveOneOut is True:
            cv = LeaveOneOut(len(label))

        if APPLY_PCA:
            processing_steps.append('PCA')
        if APPLY_OVERSAMPLING:
            processing_steps.append('Oversampling')
        processing_steps.append('Normalization')
        processing_steps.append('Downsampling')

        # %% Fold processing
        def apply_cv(epochs):
            count = 1
            confusion_matrixes = []
            confusion_matrixes_percent = []
            tn_rates = []
            tp_rates = []
            predicted = ''
            test_label = ''
            firstIterCV = True
            probabilities = np.array([[]], ndmin=2)
            predictions = np.array([])
            my_tpr_cont = []
            my_fpr_cont = []
            my_aucs = []
            best_threshold = []
            cv_probabilities = []
            cv_probabilities_label = []
            for train, test in cv:
                ## Train Data processing ##
                train_data = epochs._data[train]
                train_label = label[train]

                # Online simulation flag
                if SIMULATE_ONLINE is True:  # epochs should have one epoch only
                    train_bp = mne.filter.band_pass_filter(train_data, sfreq, l_freq, h_freq,
                                                           method='iir')  # bandpass on one epoch
                else:
                    train_bp = train_data

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
                                              trainScaleFactor, pca)
                #			test_pcaed = compute_features(test_data,sfreq,l_freq,h_freq,decim_factor,trainShiftFactor,trainScaleFactor,pca=None)

                ## Test ##
                train_x = train_pcaed
                test_x = test_pcaed

                # oversampling the least present sample
                # if APPLY_OVERSAMPLING:
                #	idx_offset = balance_idx(train_label)
                #	oversampled_train_label = np.append(train_label,train_label[idx_offset])
                #	oversampled_train_x = np.concatenate((train_x,train_x[idx_offset]),0)
                #	train_label = oversampled_train_label
                #	train_x = oversampled_train_x

                # Classifier init
                RF = dict(trees=100, maxdepth=None)
                cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'],
                                             n_jobs=n_jobs)
                # cls = RandomForestClassifier(n_estimators=RF['trees'], max_features='auto', max_depth=RF['maxdepth'], class_weight="balanced", n_jobs=n_jobs)
                # cls = LDA(solver='eigen')
                #			cls = QDA(reg_param=0.3) # regularized LDA

                #			cls.fit( train_x, train_label )
                # Y_pred= cls.predict( test_x )
                # prediction = Y_pred

                # Fitting
                #				cls= rLDA(regcoeff)
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

                # print('CV #'+str(count))
                #				print('Prediction: '+str(prediction))
                #				print('    Actual: '+str(test_label))

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
                #				print('===================')

                ## Manual ROC curve computation
                #			if useLeaveOneOut is not True:
                #				probs_np = np.array(probs)
                #				myfpr = []
                #				mytpr = []
                #				mythresh = []
                #				for thresh in np.linspace(0,1,100):
                #					newpred = [4 if x[0] < thresh else 3 for x in probs_np] #list comprehension to quickly go through the list. x[0] because hp_probs is shape (2,20)
                #					newcm = confusion_matrix(test_label,newpred)
                #					newcm_norm = newcm.astype('float') / newcm.sum(axis=1)[:, np.newaxis]
                #					mytpr.append(newcm_norm[0][0])
                #					myfpr.append(newcm_norm[1][0])
                #					mythresh.append(thresh)
                #
                #				my_tpr_cont.append(mytpr)
                #				my_fpr_cont.append(myfpr)
                #
                #				myroc_auc = auc(myfpr, mytpr)
                #				my_aucs.append(myroc_auc)

                ## One CV done, go to the next one
                count += 1

            # if useLeaveOneOut is not True:
            #			my_fpr_cont_np = np.array(my_fpr_cont)
            #			my_tpr_cont_np = np.array(my_tpr_cont)
            #
            #			my_fpr_cont_avg = np.mean(my_fpr_cont_np,axis=0)
            #			my_tpr_cont_avg = np.mean(my_tpr_cont_np,axis=0)
            #
            #
            #			plt.plot(my_fpr_cont_avg,my_tpr_cont_avg)
            #			plt.xlabel('false positive rate')
            #			plt.ylabel('true positive rate')
            #
            #
            #			auc_from_avg = auc(my_fpr_cont_avg,my_tpr_cont_avg)
            #			auc_from_my_aucs = np.mean(my_aucs)
            #
            #			# Make a subset of data where FPR < 0.2
            #			idx_below_fpr_0_2 = np.where(my_fpr_cont_avg < MAX_FPR)
            #			fpr_below_fpr_0_2 = my_fpr_cont_avg[idx_below_fpr_0_2]
            #			tpr_below_fpr_0_2 = my_tpr_cont_avg[idx_below_fpr_0_2]
            #
            #			# Look for the best (max value) FPR in that subset
            #			best_tpr_below_fpr_0_2 = np.max(tpr_below_fpr_0_2)
            #			# ... get its idx
            #			best_tpr_below_fpr_0_2_idx = np.array(np.where(my_tpr_cont_avg == best_tpr_below_fpr_0_2)).ravel()
            #
            #			# Get the associated TPRs
            #			best_tpr_below_fpr_0_2_associated_fpr = np.array(my_fpr_cont_avg)[best_tpr_below_fpr_0_2_idx]
            #			# Get the best (min value) in that subset
            #			best_associated_fpr = np.min(best_tpr_below_fpr_0_2_associated_fpr)
            #			# ... get its idx
            #			best_associated_fpr_idx = np.array(np.where(my_fpr_cont_avg == best_associated_fpr)).ravel()
            #
            #			# The best idx is the one that is on both set
            #			best_idx = best_tpr_below_fpr_0_2_idx[np.in1d(best_tpr_below_fpr_0_2_idx,best_associated_fpr_idx)]
            #			plt.xlabel('False positive rate')
            #			plt.ylabel('True positive rate')
            #			threshold_list = np.linspace(0,1,100)
            #			best_threshold = threshold_list[best_idx]
            #			print('Best treshold(s):'+str(best_threshold))
            #			print('Gives a TPR of '+str(best_tpr_below_fpr_0_2))
            #			print('And a FPR of '+str(best_associated_fpr))
            #
            #
            ##		from mpl_toolkits.mplot3d import Axes3D
            ##		fig = plt.figure()
            ##		ax = fig.add_subplot(111,projection='3d')
            ##		ax.plot(my_fpr_cont_avg,my_fpr_cont_avg,mythresh)
            ##		mean_tpr /= len(cv)
            ##		mean_tpr[-1] = 1.0
            ##		mean_auc = auc(mean_fpr, mean_tpr)
            #		#plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
            #
            ##		if useLeaveOneOut is True:
            ##			finalCM = confusion_matrix(predictions,label)
            ##			#Used w/ one out
            ##			cms = []
            ##			cms_norm = []
            ##			#threshold search
            ##			probabilities_right = probabilities[0,:]
            ##			probabilities_wrong = probabilities[1,:]
            ##			for thresh in np.arange(0,1,0.05):
            ##				pred_tmp = np.array([])
            ##				for prob in probabilities:
            ##					if prob[0] < thresh:
            ##						pred_tmp = np.append(pred_tmp,4)
            ##					else:
            ##						pred_tmp = np.append(pred_tmp,3)
            ##				cm_tmp = confusion_matrix(pred_tmp,label)
            ##				cms.append(cm_tmp)
            ##				cm_tmp_norm = cm_tmp.astype('float') / cm_tmp.sum(axis=1)[:, np.newaxis]
            ##				cms_norm.append(cm_tmp_norm)
            #		if useLeaveOneOut is True:
            #			avg_confusion_matrixes = 0
            #			auc_from_avg = 0
            #			auc_from_my_aucs = 0
            #			label_np = np.array(label)
            #			lvofpr = []
            #			lvotpr = []
            #			lvothresh = []
            #			lvocms = []
            #			threshold_list = np.linspace(0,1,100)
            #			for thresh in threshold_list:
            #				lvopred = [4 if x[0] < thresh else 3 for x in probabilities] #list comprehension to quickly go through the list. x[0] because hp_probs is shape (2,20)
            #				lvocm = confusion_matrix(label_np,lvopred)
            #				lvocm_norm = lvocm.astype('float') / lvocm.sum(axis=1)[:, np.newaxis]
            #				lvocms.append(lvocm_norm)
            #				lvotpr.append(lvocm_norm[0][0])
            #				lvofpr.append(lvocm_norm[1][0])
            #				lvothresh.append(thresh)
            #
            #			lvo_auc = auc(lvofpr,lvotpr)
            #
            #			# Make a subset of data where FPR < 0.2
            #
            #			idx_below_fpr_0_2 = np.where(np.array(lvofpr) < MAX_FPR)
            #			fpr_below_fpr_0_2 = np.array(lvofpr)[idx_below_fpr_0_2[0]]
            #			tpr_below_fpr_0_2 = np.array(lvotpr)[idx_below_fpr_0_2[0]]
            #
            #			# Look for the best (max value) FPR in that subset
            #			best_tpr_below_fpr_0_2 = np.max(tpr_below_fpr_0_2)
            #			# ... get its idx
            #			best_tpr_below_fpr_0_2_idx = np.array(np.where(lvotpr == best_tpr_below_fpr_0_2)).ravel()
            #
            #			# Get the associated TPRs
            #			best_tpr_below_fpr_0_2_associated_fpr = np.array(lvofpr)[best_tpr_below_fpr_0_2_idx]
            #			# Get the best (min value) in that subset
            #			best_associated_fpr = np.min(best_tpr_below_fpr_0_2_associated_fpr)
            #			# ... get its idx
            #			best_associated_fpr_idx = np.array(np.where(lvofpr == best_associated_fpr)).ravel()
            #
            #			# The best idx is the one that is on both set
            #			best_idx = best_tpr_below_fpr_0_2_idx[np.in1d(best_tpr_below_fpr_0_2_idx,best_associated_fpr_idx)]
            #
            #			plt.plot(lvofpr,lvotpr)
            #			plt.xlabel('False positive rate')
            #			plt.ylabel('True positive rate')
            #			best_threshold = threshold_list[best_idx]
            #			print('Best treshold:'+str(best_threshold))
            #			print('Gives a TPR of '+str(best_tpr_below_fpr_0_2))
            #			print('And a FPR of '+str(best_associated_fpr))
            #			print('CM')
            #			print(lvocms[best_idx[0]])
            auc_from_avg = None
            auc_from_my_aucs = None
            best_threshold = None
            cv_probabilities_np = np.array(cv_probabilities)
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
            best_threshold = threshold_list[best_idx]

            if False:
                plt.plot(biglist_fpr, biglist_tpr)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                print('#################################')
                print('Best treshold:' + str(best_threshold))
                print('Gives a TPR of ' + str(best_tpr_below_maxfpr))
                print('And a FPR of ' + str(best_associated_fpr))
                print('CM')
                print(biglist_cms[best_idx[0]])
            return (biglist_auc, biglist_cms, best_threshold, best_tpr_below_maxfpr)

        biglist_auc, biglist_cms, best_threshold = apply_cv(epochs)

        return (biglist_auc, biglist_cms, best_threshold, best_tpr_below_maxfpr)


    #	for vartmax in np.arange(0.5,2,0.1):
    #		print('Testing'+str(vartmax))
    #		tmin = 0
    #		tmax = vartmax
    #		avg_confusion_matrixes,my_auc = getperf(APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV,\
    #										l_freq,h_freq, picks_feat, offset, tmin, tmax,\
    #										baselineRange,reg_coeff=0.3, verbose=False)
    #		params.append([[APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV, l_freq,h_freq, picks_feat, offset, tmin, tmax,baselineRange,0.3, False]])
    #		results.append([avg_confusion_matrixes,my_auc])
    #
    #	tmax = 1
    #	params_regcoeff = []
    #	results_regcoeff = []
    #	for varregcoeff in np.arange(0.0,1,0.1):
    #		print('Testing '+str(varregcoeff))
    #		avg_confusion_matrixes, avg_auc,my_auc = getperf(APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV,\
    #										l_freq,h_freq, picks_feat, offset, tmin, tmax,\
    #										baselineRange,reg_coeff=varregcoeff, verbose=False)
    #		params_regcoeff.append([[APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV, l_freq,h_freq, picks_feat, offset, tmin, tmax,baselineRange,0.3, False]])
    #		results_regcoeff.append([avg_confusion_matrixes,my_auc])
    params = []
    results = []
    cmats = []  # this should be discarded if you activate leaveOneOut
    threshlist = []
    tprlist = []
    var_tmax_range = np.arange(0.1, 1.0, 0.4)
    var_regcoeff_range = np.arange(0.2, 1.0, 0.4)
    APPLY_CAR = False
    APPLY_PCA = False
    for var_tmax in var_tmax_range:
        results_regcoeff = []
        param_regcoeff = []
        cmats_regcoeff = []
        bthresh_regcoeff = []
        btpr_regcoeff = []
        for var_regcoeff in var_regcoeff_range:
            print('testing [' + str(var_tmax) + ',' + str(var_regcoeff) + ']')
            tmax = var_tmax
            regcoeff = var_regcoeff
            biglist_auc, biglist_cms, best_threshold, besttpr = getperf(APPLY_CAR, APPLY_PCA, APPLY_OVERSAMPLING, DO_CV,\
                                                                        l_freq, h_freq, picks_feat, offset, tmin, tmax,\
                                                                        baselineRange, reg_coeff=regcoeff,
                                                                        verbose=False)
            results_regcoeff.append(biglist_auc)
            cmats_regcoeff.append(biglist_cms)
            param_regcoeff.append([tmax, regcoeff])
            bthresh_regcoeff.append(best_threshold)
            btpr_regcoeff.append(besttpr)
            print('done testing')
        results.append(results_regcoeff)
        cmats.append(cmats_regcoeff)
        params.append(param_regcoeff)
        threshlist.append(bthresh_regcoeff)
        tprlist.append(done)
    results_np = np.array(results)
    params_np = np.array(params)
    cmats_np = np.array(cmats)
    threshlist_np = np.array(threshlist)
    tprlist_np = np.array(tprlist)

    best_result = np.max(tprlist_np)
    best_results_idxs = np.where(tprlist_np == tprlist)
    best_results_cmat = cmats_np[
        best_results_idxs[0][0], best_results_idxs[1][0]]  # take the first best results because... no reasons
    best_results_param = params_np[best_results_idxs[0][0], best_results_idxs[1][0]]

    plt.matshow(tprlist)
    plt.xticks(range(0, len(var_regcoeff_range)), var_regcoeff_range)
    plt.yticks(range(0, len(var_tmax_range)), var_tmax_range)
    plt.xlabel('Reg Coeff')
    plt.ylabel('Tmax length')

    print('Best results:', best_result)

# print('############################')
#	print('Summary')
#	print('-------')
#	#print('Processing steps: '+ str(processing_steps))
#	print('Average Confusion matrix')
#	print(avg_confusion_matrixes1)
#	print('Average true positive rate: '+str(avg_tp_rates1)) # prediction of "wrong" as "wrong"
#	print('Average true negative rate: '+str(avg_tn_rates1)) # prediction of "correct" as "correct"
#	print('Mean AUC: '+str(avg_auc1))

#	if OUTPUT_PCL:
#		ch_names= [raw.info['ch_names'][c] for c in picks_feat]
#
#		train_data = epochs._data
#		train_bp= mne.filter.band_pass_filter(train_data, sfreq, l_freq, h_freq, method='iir' ) #bandpass
#		(train_data_normalized,trainShiftFactor,trainScaleFactor) = normalizeAcrossEpoch(train_bp,'MinMax')
#		train_data_downsampled = train_data_normalized[:,:,::decim_factor]
#		train_reshaped = train_data_downsampled.reshape(train_data_downsampled.shape[0],-1) # merge channel and time for the pca
#		pca = PCA(0.95)
#		pca.fit(train_reshaped)
#		pca.components_ = -pca.components_
#		train_pcaed= pca.transform( train_reshaped )
#		w,b = trainLDA(train_pcaed, label, 0.3)
#
#		data= dict(sfreq=raw.info['sfreq'], ch_names=ch_names, picks=picks_feat, \
#				w=w,b=b, l_freq=l_freq, h_freq=h_freq, decim_factor=decim_factor, pca=pca,
#				shiftFactor=trainShiftFactor, scaleFactor=trainScaleFactor)
#		outdir= DATADIR + '/errp_classifier'
#		qc.make_dirs(outdir)
#		clsfile= outdir + '/errp_classifier.pcl'
#		qc.save_obj(clsfile, data)
#		print('Saved as %s' % clsfile)
# print('Done')






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
