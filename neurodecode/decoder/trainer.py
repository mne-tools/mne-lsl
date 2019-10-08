from __future__ import print_function, division

"""
trainer.py

Perform cross-validation and train a classifier.
See run() to see the flow.


Kyuhwa Lee, 2018
Swiss Federal Institute of Technology Lausanne (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import sys
import imp
import mne
import mne.io
import platform
import numpy as np
import multiprocessing as mp
import sklearn.metrics as skmetrics
import neurodecode.utils.q_common as qc
import neurodecode.utils.pycnbi_utils as pu
import neurodecode.decoder.features as features
from builtins import input
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from neurodecode import logger
from neurodecode.decoder.rlda import rLDA
from neurodecode.triggers.trigger_def import trigger_def
from neurodecode.gui.streams import redirect_stdout_to_queue

# scikit-learn old version compatibility
try:
    from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut
    SKLEARN_OLD = False
except ImportError:
    from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
    SKLEARN_OLD = True
mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

def check_config(cfg):
    critical_vars = {
        'COMMON': ['TRIGGER_FILE',
                    'TRIGGER_DEF',
                    'EPOCH',
                    'DATA_PATH',
                    'PICKED_CHANNELS',
                    'SP_FILTER',
                    'SP_CHANNELS',
                    'TP_FILTER',
                    'NOTCH_FILTER',
                    'FEATURES',
                    'CLASSIFIER',
                    'CV_PERFORM'],
                    'RF': ['trees', 'depth', 'seed'],
                    'GB': ['trees', 'learning_rate', 'depth', 'seed'],
                    'LDA': [],
                    'rLDA': ['r_coeff'],
                    'StratifiedShuffleSplit': ['test_ratio', 'folds', 'seed', 'export_result'],
                    'LeaveOneOut': ['export_result']
    }

    # optional variables with default values
    optional_vars = {
        'MULTIPLIER': 1,
        'EXPORT_GOOD_FEATURES': False,
        'FEAT_TOPN': 10,
        'EXPORT_CLS': False,
        'REREFERENCE': None,
        'N_JOBS': None,
        'EXCLUDED_CHANNELS': None,
        'LOAD_EVENTS': None,
        'CV': {'IGNORE_THRES': None, 'DECISION_THRES': None, 'BALANCE_SAMPLES': False},
    }

    for v in critical_vars['COMMON']:
        if not hasattr(cfg, v):
            logger.error('%s not defined in config.' % v)
            raise RuntimeError

    for key in optional_vars:
        if not hasattr(cfg, key):
            setattr(cfg, key, optional_vars[key])
            logger.warning('Setting undefined parameter %s=%s' % (key, getattr(cfg, key)))

    if 'decim' not in cfg.FEATURES['PSD']:
        cfg.FEATURES['PSD']['decim'] = 1

    # classifier parameters check
    selected_classifier = cfg.CLASSIFIER[cfg.CLASSIFIER['selected']]

    if selected_classifier == 'RF':
        if 'RF' not in cfg.CLASSIFIER:
            logger.error('"RF" not defined in config.')
            raise RuntimeError
        for v in critical_vars['RF']:
            if v not in cfg.CLASSIFIER['RF']:
                logger.error('%s not defined in config.' % v)
                raise RuntimeError

    elif selected_classifier == 'GB' or selected_classifier == 'XGB':
        if 'GB' not in cfg.CLASSIFIER:
            logger.error('"GB" not defined in config.')
            raise RuntimeError
        for v in critical_vars['GB']:
            if v not in cfg.CLASSIFIER[selected_classifier]:
                logger.error('%s not defined in config.' % v)
                raise RuntimeError

    elif selected_classifier == 'rLDA':
        if 'rLDA' not in cfg.CLASSIFIER:
            logger.error('"rLDA" not defined in config.')
            raise RuntimeError
        for v in critical_vars['rLDA']:
            if v not in cfg.CLASSIFIER['rLDA']:
                logger.error('%s not defined in config.' % v)
                raise RuntimeError

    cv_selected = cfg.CV_PERFORM['selected']
    if cfg.CV_PERFORM[cv_selected] is not None:
        if cv_selected == 'StratifiedShuffleSplit':
            if 'StratifiedShuffleSplit' not in cfg.CV_PERFORM:
                logger.error('"StratifiedShuffleSplit" not defined in config.')
                raise RuntimeError
            for v in critical_vars['StratifiedShuffleSplit']:
                if v not in cfg.CV_PERFORM[cv_selected]:
                    logger.error('%s not defined in config.' % v)
                    raise RuntimeError

        elif cv_selected == 'LeaveOneOut':
            if 'LeaveOneOut' not in cfg.CV_PERFORM:
                logger.error('"LeaveOneOut" not defined in config.')
                raise RuntimeError
            for v in critical_vars['LeaveOneOut']:
                if v not in cfg.CV_PERFORM[cv_selected]:
                    logger.error('%s not defined in config.' % v)
                    raise RuntimeError

    if cfg.N_JOBS is None:
        cfg.N_JOBS = mp.cpu_count()

    return cfg


def balance_samples(X, Y, balance_type, verbose=False):
    if balance_type == 'OVER':
        """
        Oversample from classes that lack samples
        """
        label_set = np.unique(Y)
        max_set = []
        X_balanced = np.array(X)
        Y_balanced = np.array(Y)

        # find a class with maximum number of samples
        for c in label_set:
            yl = np.where(Y == c)[0]
            if len(max_set) == 0 or len(yl) > max_set[1]:
                max_set = [c, len(yl)]
        for c in label_set:
            if c == max_set[0]: continue
            yl = np.where(Y == c)[0]
            extra_samples = max_set[1] - len(yl)
            extra_idx = np.random.choice(yl, extra_samples)
            X_balanced = np.append(X_balanced, X[extra_idx], axis=0)
            Y_balanced = np.append(Y_balanced, Y[extra_idx], axis=0)
    elif balance_type == 'UNDER':
        """
        Undersample from classes that are excessive
        """
        label_set = np.unique(Y)
        min_set = []

        # find a class with minimum number of samples
        for c in label_set:
            yl = np.where(Y == c)[0]
            if len(min_set) == 0 or len(yl) < min_set[1]:
                min_set = [c, len(yl)]
        yl = np.where(Y == min_set[0])[0]
        X_balanced = np.array(X[yl])
        Y_balanced = np.array(Y[yl])
        for c in label_set:
            if c == min_set[0]: continue
            yl = np.where(Y == c)[0]
            reduced_idx = np.random.choice(yl, min_set[1])
            X_balanced = np.append(X_balanced, X[reduced_idx], axis=0)
            Y_balanced = np.append(Y_balanced, Y[reduced_idx], axis=0)
    elif balance_type is None or balance_type is False:
        return X, Y
    else:
        logger.error('Unknown balancing type %s' % balance_type)
        raise ValueError

    logger.info_green('\nNumber of samples after %ssampling' % balance_type.lower())
    for c in label_set:
        logger.info('%s: %d -> %d' % (c, len(np.where(Y == c)[0]), len(np.where(Y_balanced == c)[0])))

    return X_balanced, Y_balanced


def crossval_epochs(cv, epochs_data, labels, cls, label_names=None, do_balance=None, n_jobs=None, ignore_thres=None, decision_thres=None):
    """
    Epoch-based cross-validation used by cross_validate().

    Params
    ======
    cv: scikit-learn cross-validation object
    epochs_data: np.array of shape [epochs x samples x features]
    labels: np.array of shape [epochs x samples]
    cls: classifier
    label_names: associated label names {0:'Left', 1:'Right', ...}
    do_balance: oversample or undersample to match the number of samples among classes

    """

    scores = []
    f1s = []
    cnum = 1
    cm_sum = 0
    label_set = np.unique(labels)
    num_labels = len(label_set)
    if label_names is None:
        label_names = {l:'%s' % l for l in label_set}

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if n_jobs > 1:
        logger.info('crossval_epochs(): Using %d cores' % n_jobs)
        pool = mp.Pool(n_jobs)
        results = []

    # for classifier itself, single core is usually faster
    cls.n_jobs = 1

    if SKLEARN_OLD:
        splits = cv
    else:
        splits = cv.split(epochs_data, labels[:, 0])
    for train, test in splits:
        X_train = np.concatenate(epochs_data[train])
        X_test = np.concatenate(epochs_data[test])
        Y_train = np.concatenate(labels[train])
        Y_test = np.concatenate(labels[test])
        if do_balance:
            X_train, Y_train = balance_samples(X_train, Y_train, do_balance)
            X_test, Y_test = balance_samples(X_test, Y_test, do_balance)

        if n_jobs > 1:
            results.append(pool.apply_async(fit_predict_thres,
                                            [cls, X_train, Y_train, X_test, Y_test, cnum, label_set, ignore_thres, decision_thres]))
        else:
            score, cm, f1 = fit_predict_thres(cls, X_train, Y_train, X_test, Y_test, cnum, label_set, ignore_thres, decision_thres)
            scores.append(score)
            f1s.append(f1)
            cm_sum += cm
        cnum += 1

    if n_jobs > 1:
        pool.close()
        pool.join()

        for r in results:
            score, cm, f1 = r.get()
            scores.append(score)
            f1s.append(f1)
            cm_sum += cm

    # confusion matrix
    cm_sum = cm_sum.astype('float')
    if cm_sum.shape[0] != cm_sum.shape[1]:
        # we have decision thresholding condition
        assert cm_sum.shape[0] < cm_sum.shape[1]
        cm_sum_all = cm_sum
        cm_sum = cm_sum[:, :cm_sum.shape[0]]
        underthres = np.array([r[-1] / sum(r) for r in cm_sum_all])
    else:
        underthres = None

    cm_rate = np.zeros(cm_sum.shape)
    for r_in, r_out in zip(cm_sum, cm_rate):
        rs = sum(r_in)
        if rs > 0:
            r_out[:] = r_in / rs
        else:
            assert min(r) == max(r) == 0
    if underthres is not None:
        cm_rate = np.concatenate((cm_rate, underthres[:, np.newaxis]), axis=1)

    cm_txt = 'Y: ground-truth, X: predicted\n'
    max_chars = 12
    tpl_str = '%%-%ds ' % max_chars
    tpl_float = '%%-%d.2f ' % max_chars
    for l in label_set:
        cm_txt += tpl_str % label_names[l][:max_chars]
    if underthres is not None:
        cm_txt += tpl_str % 'Ignored'
    cm_txt += '\n'
    for r in cm_rate:
        for c in r:
            cm_txt += tpl_float % c
        cm_txt += '\n'
    cm_txt += 'Average accuracy: %.2f\n' % np.mean(scores)
    cm_txt += 'Average F1 score: %.2f\n' % np.mean(f1s)

    return np.array(scores), cm_txt


def balance_tpr(cfg, featdata):
    """
    Find the threshold of class index 0 that yields equal number of true positive samples of each class.
    Currently only available for binary classes.

    Params
    ======
    cfg: config module
    feetdata: feature data computed using compute_features()
    """

    n_jobs = cfg.N_JOBS
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs > 1:
        logger.info('balance_tpr(): Using %d cores' % n_jobs)
        pool = mp.Pool(n_jobs)
        results = []

    # Init a classifier
    selected_classifier = cfg.CLASSIFIER[cfg.CLASSIFIER['selected']]
    if selected_classifier == 'GB':
        cls = GradientBoostingClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER['GB']['learning_rate'],
                                         n_estimators=cfg.CLASSIFIER['GB']['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER['GB']['depth'],
                                         random_state=cfg.CLASSIFIER[selected_classifier]['seed'], max_features='sqrt', verbose=0, warm_start=False,
                                         presort='auto')
    elif selected_classifier == 'XGB':
        cls = XGBClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER['XGB']['learning_rate'],
                                         n_estimators=cfg.CLASSIFIER['XGB']['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER['XGB']['depth'],
                                         random_state=cfg.CLASSIFIER['XGB']['seed'], max_features='sqrt', verbose=0, warm_start=False,
                                         presort='auto')
    elif selected_classifier == 'RF':
        cls = RandomForestClassifier(n_estimators=cfg.CLASSIFIER['RF']['trees'], max_features='auto',
                                     max_depth=cfg.CLASSIFIER['RF']['depth'], n_jobs=cfg.N_JOBS, random_state=cfg.CLASSIFIER['RF']['seed'],
                                     oob_score=False, class_weight='balanced_subsample')
    elif selected_classifier == 'LDA':
        cls = LDA()
    elif selected_classifier == 'rLDA':
        cls = rLDA(cfg.CLASSIFIER['rLDA'])
    else:
        logger.error('Unknown classifier type %s' % selected_classifier)
        raise ValueError

    # Setup features
    X_data = featdata['X_data']
    Y_data = featdata['Y_data']
    wlen = featdata['wlen']
    if cfg.CLASSIFIER['PSD']['wlen'] is None:
        cfg.CLASSIFIER['PSD']['wlen'] = wlen

    # Choose CV type
    ntrials, nsamples, fsize = X_data.shape
    selected_CV = cfg.CV_PERFORM[cfg.CV_PERFORM['selected']]
    if cselected_CV == 'LeaveOneOut':
        logger.info_green('\n%d-fold leave-one-out cross-validation' % ntrials)
        if SKLEARN_OLD:
            cv = LeaveOneOut(len(Y_data))
        else:
            cv = LeaveOneOut()
    elif selected_CV == 'StratifiedShuffleSplit':
        logger.info_green('\n%d-fold stratified cross-validation with test set ratio %.2f' % (cfg.CV_PERFORM[selected_CV]['folds'], cfg.CV_PERFORM[selected_CV]['test_ratio']))
        if SKLEARN_OLD:
            cv = StratifiedShuffleSplit(Y_data[:, 0], cfg.CV_PERFORM[selected_CV]['folds'], test_size=cfg.CV_PERFORM[selected_CV]['test_ratio'], random_state=cfg.CV_PERFORM[selected_CV]['random_seed'])
        else:
            cv = StratifiedShuffleSplit(n_splits=cfg.CV_PERFORM[selected_CV]['folds'], test_size=cfg.CV_PERFORM[selected_CV]['test_ratio'], random_state=cfg.CV_PERFORM[selected_CV]['random_seed'])
    else:
        logger.error('%s is not supported yet. Sorry.' % selected_CV)
        raise NotImplementedError
    logger.info('%d trials, %d samples per trial, %d feature dimension' % (ntrials, nsamples, fsize))

    # For classifier itself, single core is usually faster
    cls.n_jobs = 1
    Y_preds = []

    if SKLEARN_OLD:
        splits = cv
    else:
        splits = cv.split(X_data, Y_data[:, 0])
    for cnum, (train, test) in enumerate(splits):
        X_train = np.concatenate(X_data[train])
        X_test = np.concatenate(X_data[test])
        Y_train = np.concatenate(Y_data[train])
        Y_test = np.concatenate(Y_data[test])
        if n_jobs > 1:
            results.append(pool.apply_async(get_predict_proba, [cls, X_train, Y_train, X_test, Y_test, cnum+1]))
        else:
            Y_preds.append(get_predict_proba(cls, X_train, Y_train, X_test, Y_test, cnum+1))
        cnum += 1

    # Aggregate predictions
    if n_jobs > 1:
        pool.close()
        pool.join()
        for r in results:
            Y_preds.append(r.get())
    Y_preds = np.concatenate(Y_preds, axis=0)

    # Find threshold for class index 0
    Y_preds = sorted(Y_preds)
    mid_idx = int(len(Y_preds) / 2)
    if len(Y_preds) == 1:
        return 0.5 # should not reach here in normal conditions
    elif len(Y_preds) % 2 == 0:
        thres = Y_preds[mid_idx-1] + (Y_preds[mid_idx] - Y_preds[mid_idx-1]) / 2
    else:
        thres = Y_preds[mid_idx]
    return thres


def cva_features(datadir):
    """
    (DEPRECATED FUNCTION)
    """
    for fin in qc.get_file_list(datadir, fullpath=True):
        if fin[-4:] != '.gdf': continue
        fout = fin + '.cva'
        if os.path.exists(fout):
            logger.info('Skipping', fout)
            continue
        logger.info("cva_features('%s')" % fin)
        qc.matlab("cva_features('%s')" % fin)


def get_predict_proba(cls, X_train, Y_train, X_test, Y_test, cnum):
    """
    All likelihoods will be collected from every fold of a cross-validaiton. Based on these likelihoods,
    a threshold will be computed that will balance the true positive rate of each class.
    Available with binary classification scenario only.
    """
    timer = qc.Timer()
    cls.fit(X_train, Y_train)
    Y_pred = cls.predict_proba(X_test)
    logger.info('Cross-validation %d (%d tests) - %.1f sec' % (cnum, Y_pred.shape[0], timer.sec()))
    return Y_pred[:,0]


def fit_predict_thres(cls, X_train, Y_train, X_test, Y_test, cnum, label_list, ignore_thres=None, decision_thres=None):
    """
    Any likelihood lower than a threshold is not counted as classification score
    Confusion matrix, accuracy and F1 score (macro average) are computed.

    Params
    ======
    ignore_thres:
    if not None or larger than 0, likelihood values lower than ignore_thres will be ignored
    while computing confusion matrix.

    """
    timer = qc.Timer()
    cls.fit(X_train, Y_train)
    assert ignore_thres is None or ignore_thres >= 0
    if ignore_thres is None or ignore_thres == 0:
        Y_pred = cls.predict(X_test)
        score = skmetrics.accuracy_score(Y_test, Y_pred)
        cm = skmetrics.confusion_matrix(Y_test, Y_pred, label_list)
        f1 = skmetrics.f1_score(Y_test, Y_pred, average='macro')
    else:
        if decision_thres is not None:
            logger.error('decision threshold and ignore_thres cannot be set at the same time.')
            raise ValueError
        Y_pred = cls.predict_proba(X_test)
        Y_pred_labels = np.argmax(Y_pred, axis=1)
        Y_pred_maxes = np.array([x[i] for i, x in zip(Y_pred_labels, Y_pred)])
        Y_index_overthres = np.where(Y_pred_maxes >= ignore_thres)[0]
        Y_index_underthres = np.where(Y_pred_maxes < ignore_thres)[0]
        Y_pred_overthres = np.array([cls.classes_[x] for x in Y_pred_labels[Y_index_overthres]])
        Y_pred_underthres = np.array([cls.classes_[x] for x in Y_pred_labels[Y_index_underthres]])
        Y_pred_underthres_count = np.array([np.count_nonzero(Y_pred_underthres == c) for c in label_list])
        Y_test_overthres = Y_test[Y_index_overthres]
        score = skmetrics.accuracy_score(Y_test_overthres, Y_pred_overthres)
        cm = skmetrics.confusion_matrix(Y_test_overthres, Y_pred_overthres, label_list)
        cm = np.concatenate((cm, Y_pred_underthres_count[:, np.newaxis]), axis=1)
        f1 = skmetrics.f1_score(Y_test_overthres, Y_pred_overthres, average='macro')

    logger.info('Cross-validation %d (%.3f) - %.1f sec' % (cnum, score, timer.sec()))
    return score, cm, f1


def cross_validate(cfg, featdata, cv_file=None):
    """
    Perform cross validation
    """
    # Init a classifier
    selected_classifier = cfg.CLASSIFIER['selected']
    if selected_classifier == 'GB':
        cls = GradientBoostingClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER['GB']['learning_rate'], presort='auto',
                                         n_estimators=cfg.CLASSIFIER['GB']['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER['GB']['depth'],
                                         random_state=cfg.CLASSIFIER['GB']['seed'], max_features='sqrt', verbose=0, warm_start=False)
    elif selected_classifier == 'XGB':
        cls = XGBClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER['XGB']['learning_rate'], presort='auto',
                                         n_estimators=cfg.CLASSIFIER['XGB']['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER['XGB']['depth'],
                                         random_state=cfg.CLASSIFIER['XGB'], max_features='sqrt', verbose=0, warm_start=False)
    elif selected_classifier == 'RF':
        cls = RandomForestClassifier(n_estimators=cfg.CLASSIFIER['RF']['trees'], max_features='auto',
                                     max_depth=cfg.CLASSIFIER['RF']['depth'], n_jobs=cfg.N_JOBS, random_state=cfg.CLASSIFIER['RF']['seed'],
                                     oob_score=False, class_weight='balanced_subsample')
    elif selected_classifier == 'LDA':
        cls = LDA()
    elif selected_classifier == 'rLDA':
        cls = rLDA(cfg.CLASSIFIER['rLDA']['r_coeff'])
    else:
        logger.error('Unknown classifier type %s' % selected_classifier)
        raise ValueError

    # Setup features
    X_data = featdata['X_data']
    Y_data = featdata['Y_data']
    wlen = featdata['wlen']

    # Choose CV type
    ntrials, nsamples, fsize = X_data.shape
    selected_cv =  cfg.CV_PERFORM['selected']
    if selected_cv == 'LeaveOneOut':
        logger.info_green('%d-fold leave-one-out cross-validation' % ntrials)
        if SKLEARN_OLD:
            cv = LeaveOneOut(len(Y_data))
        else:
            cv = LeaveOneOut()
    elif selected_cv == 'StratifiedShuffleSplit':
        logger.info_green('%d-fold stratified cross-validation with test set ratio %.2f' % (cfg.CV_PERFORM[selected_cv]['folds'], cfg.CV_PERFORM[selected_cv]['test_ratio']))
        if SKLEARN_OLD:
            cv = StratifiedShuffleSplit(Y_data[:, 0], cfg.CV_PERFORM[selected_cv]['folds'], test_size=cfg.CV_PERFORM[selected_cv]['test_ratio'], random_state=cfg.CV_PERFORM[selected_cv]['seed'])
        else:
            cv = StratifiedShuffleSplit(n_splits=cfg.CV_PERFORM[selected_cv]['folds'], test_size=cfg.CV_PERFORM[selected_cv]['test_ratio'], random_state=cfg.CV_PERFORM[selected_cv]['seed'])
    else:
        logger.error('%s is not supported yet. Sorry.' % cfg.CV_PERFORM[cfg.CV_PERFORM['selected']])
        raise NotImplementedError
    logger.info('%d trials, %d samples per trial, %d feature dimension' % (ntrials, nsamples, fsize))

    # Do it!
    timer_cv = qc.Timer()
    scores, cm_txt = crossval_epochs(cv, X_data, Y_data, cls, cfg.tdef.by_value, cfg.CV['BALANCE_SAMPLES'], n_jobs=cfg.N_JOBS,
                                     ignore_thres=cfg.CV['IGNORE_THRES'], decision_thres=cfg.CV['DECISION_THRES'])
    t_cv = timer_cv.sec()

    # Export results
    txt = 'Cross validation took %d seconds.\n' % t_cv
    txt += '\n- Class information\n'
    txt += '%d epochs, %d samples per epoch, %d feature dimension (total %d samples)\n' %\
        (ntrials, nsamples, fsize, ntrials * nsamples)
    for ev in np.unique(Y_data):
        txt += '%s: %d trials\n' % (cfg.tdef.by_value[ev], len(np.where(Y_data[:, 0] == ev)[0]))
    if cfg.CV['BALANCE_SAMPLES']:
        txt += 'The number of samples was balanced using %ssampling.\n' % cfg.BALANCE_SAMPLES.lower()
    txt += '\n- Experiment condition\n'
    txt += 'Sampling frequency: %.3f Hz\n' % featdata['sfreq']
    txt += 'Spatial filter: %s (channels: %s)\n' % (cfg.SP_FILTER, cfg.SP_CHANNELS)
    txt += 'Spectral filter: %s\n' % cfg.TP_FILTER[cfg.TP_FILTER['selected']]
    txt += 'Notch filter: %s\n' % cfg.NOTCH_FILTER[cfg.NOTCH_FILTER['selected']]
    txt += 'Channels: ' + ','.join([str(featdata['ch_names'][p]) for p in featdata['picks']]) + '\n'
    txt += 'PSD range: %.1f - %.1f Hz\n' % (cfg.FEATURES['PSD']['fmin'], cfg.FEATURES['PSD']['fmax'])
    txt += 'Window step: %.2f msec\n' % (1000.0 * cfg.FEATURES['PSD']['wstep'] / featdata['sfreq'])
    if type(wlen) is list:
        for i, w in enumerate(wlen):
            txt += 'Window size: %.1f msec\n' % (w * 1000.0)
            txt += 'Epoch range: %s sec\n' % (cfg.EPOCH[i])
    else:
        txt += 'Window size: %.1f msec\n' % (cfg.FEATURES['PSD']['wlen'] * 1000.0)
        txt += 'Epoch range: %s sec\n' % (cfg.EPOCH)
    txt += 'Decimation factor: %d\n' % cfg.FEATURES['PSD']['decim']

    # Compute stats
    cv_mean, cv_std = np.mean(scores), np.std(scores)
    txt += '\n- Average CV accuracy over %d epochs (random seed=%s)\n' % (ntrials, cfg.CV_PERFORM[cfg.CV_PERFORM['selected']]['seed'])
    if cfg.CV_PERFORM[cfg.CV_PERFORM['selected']] in ['LeaveOneOut', 'StratifiedShuffleSplit']:
        txt += "mean %.3f, std: %.3f\n" % (cv_mean, cv_std)
    txt += 'Classifier: %s, ' % selected_classifier
    if selected_classifier == 'RF':
        txt += '%d trees, %s max depth, random state %s\n' % (
            cfg.CLASSIFIER['RF']['trees'], cfg.CLASSIFIER['RF']['depth'], cfg.CLASSIFIER['RF']['seed'])
    elif selected_classifier == 'GB' or selected_classifier == 'XGB':
        txt += '%d trees, %s max depth, %s learing_rate, random state %s\n' % (
            cfg.CLASSIFIER['GB']['trees'], cfg.CLASSIFIER['GB']['depth'], cfg.CLASSIFIER['GB']['learning_rate'], cfg.CLASSIFIER['GB']['seed'])
    elif selected_classifier == 'rLDA':
        txt += 'regularization coefficient %.2f\n' % cfg.CLASSIFIER['rLDA']['r_coeff']
    if cfg.CV['IGNORE_THRES'] is not None:
        txt += 'Decision threshold: %.2f\n' % cfg.CV['IGNORE_THRES']
    txt += '\n- Confusion Matrix\n' + cm_txt
    logger.info(txt)

    # Export to a file
    if 'export_result' in cfg.CV_PERFORM[selected_cv] and cfg.CV_PERFORM[selected_cv]['export_result'] is True:
        if cv_file is None:
            if cfg.EXPORT_CLS is True:
                qc.make_dirs('%s/classifier' % cfg.DATA_PATH)
                fout = open('%s/classifier/cv_result.txt' % cfg.DATA_PATH, 'w')
            else:
                fout = open('%s/cv_result.txt' % cfg.DATA_PATH, 'w')
        else:
            fout = open(cv_file, 'w')
        fout.write(txt)
        fout.close()


def train_decoder(cfg, featdata, feat_file=None):
    """
    Train the final decoder using all data
    """
    # Init a classifier
    selected_classifier = cfg.CLASSIFIER['selected']
    if selected_classifier == 'GB':
        cls = GradientBoostingClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER[selected_classifier]['learning_rate'],
                                         n_estimators=cfg.CLASSIFIER[selected_classifier]['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER[selected_classifier]['depth'],
                                         random_state=cfg.CLASSIFIER[selected_classifier]['seed'], max_features='sqrt', verbose=0, warm_start=False,
                                         presort='auto')
    elif selected_classifier == 'XGB':
        cls = XGBClassifier(loss='deviance', learning_rate=cfg.CLASSIFIER[selected_classifier]['learning_rate'],
                                         n_estimators=cfg.CLASSIFIER[selected_classifier]['trees'], subsample=1.0, max_depth=cfg.CLASSIFIER[selected_classifier]['depth'],
                                         random_state=cfg.GB['seed'], max_features='sqrt', verbose=0, warm_start=False,
                                         presort='auto')
    elif selected_classifier == 'RF':
        cls = RandomForestClassifier(n_estimators=cfg.CLASSIFIER[selected_classifier]['trees'], max_features='auto',
                                     max_depth=cfg.CLASSIFIER[selected_classifier]['depth'], n_jobs=cfg.N_JOBS, random_state=cfg.CLASSIFIER[selected_classifier]['seed'],
                                     oob_score=False, class_weight='balanced_subsample')
    elif selected_classifier == 'LDA':
        cls = LDA()
    elif selected_classifier == 'rLDA':
        cls = rLDA(cfg.CLASSIFIER[selected_classifier][r_coeff])
    else:
        logger.error('Unknown classifier %s' % selected_classifier)
        raise ValueError

    # Setup features
    X_data = featdata['X_data']
    Y_data = featdata['Y_data']
    wlen = featdata['wlen']
    if cfg.FEATURES['PSD']['wlen'] is None:
        cfg.FEATURES['PSD']['wlen'] = wlen
    w_frames = featdata['w_frames']
    ch_names = featdata['ch_names']
    X_data_merged = np.concatenate(X_data)
    Y_data_merged = np.concatenate(Y_data)
    if cfg.CV['BALANCE_SAMPLES']:
        X_data_merged, Y_data_merged = balance_samples(X_data_merged, Y_data_merged, cfg.CV['BALANCE_SAMPLES'], verbose=True)

    # Start training the decoder
    logger.info_green('Training the decoder')
    timer = qc.Timer()
    cls.n_jobs = cfg.N_JOBS
    cls.fit(X_data_merged, Y_data_merged)
    logger.info('Trained %d samples x %d dimension in %.1f sec' %\
          (X_data_merged.shape[0], X_data_merged.shape[1], timer.sec()))
    cls.n_jobs = 1 # always set n_jobs=1 for testing

    # Export the decoder
    classes = {c:cfg.tdef.by_value[c] for c in np.unique(Y_data)}
    if cfg.FEATURES['selected'] == 'PSD':
        data = dict(cls=cls, ch_names=ch_names, psde=featdata['psde'], sfreq=featdata['sfreq'],
                    picks=featdata['picks'], classes=classes, epochs=cfg.EPOCH, w_frames=w_frames,
                    w_seconds=cfg.FEATURES['PSD']['wlen'], wstep=cfg.FEATURES['PSD']['wstep'], spatial=cfg.SP_FILTER,
                    spatial_ch=featdata['picks'], spectral=cfg.TP_FILTER[cfg.TP_FILTER['selected']], spectral_ch=featdata['picks'],
                    notch=cfg.NOTCH_FILTER[cfg.NOTCH_FILTER['selected']], notch_ch=featdata['picks'], multiplier=cfg.MULTIPLIER,
                    ref_ch=cfg.REREFERENCE[cfg.REREFERENCE['selected']], decim=cfg.FEATURES['PSD']['decim'])
    clsfile = '%s/classifier/classifier-%s.pkl' % (cfg.DATA_PATH, platform.architecture()[0])
    qc.make_dirs('%s/classifier' % cfg.DATA_PATH)
    qc.save_obj(clsfile, data)
    logger.info('Decoder saved to %s' % clsfile)

    # Reverse-lookup frequency from FFT
    fq = 0
    if type(cfg.FEATURES['PSD']['wlen']) == list:
        fq_res = 1.0 / cfg.FEATURES['PSD']['wlen'][0]
    else:
        fq_res = 1.0 / cfg.FEATURES['PSD']['wlen']
    fqlist = []
    while fq <= cfg.FEATURES['PSD']['fmax']:
        if fq >= cfg.FEATURES['PSD']['fmin']:
            fqlist.append(fq)
        fq += fq_res

    # Show top distinctive features
    if cfg.FEATURES['selected'] == 'PSD':
        logger.info_green('Good features ordered by importance')
        if selected_classifier in ['RF', 'GB', 'XGB']:
            keys, values = qc.sort_by_value(list(cls.feature_importances_), rev=True)
        elif selected_classifier in ['LDA', 'rLDA']:
            keys, values = qc.sort_by_value(cls.w, rev=True)
        keys = np.array(keys)
        values = np.array(values)

        if cfg.EXPORT_GOOD_FEATURES:
            if feat_file is None:
                gfout = open('%s/classifier/good_features.txt' % cfg.DATA_PATH, 'w')
            else:
                gfout = open(feat_file, 'w')

        if type(wlen) is not list:
            ch_names = [ch_names[c] for c in featdata['picks']]
        else:
            ch_names = []
            for w in range(len(wlen)):
                for c in featdata['picks']:
                    ch_names.append('w%d-%s' % (w, ch_names[c]))

        chlist, hzlist = features.feature2chz(keys, fqlist, ch_names=ch_names)
        valnorm = values[:cfg.FEAT_TOPN].copy()
        valsum = np.sum(valnorm)
        if valsum == 0:
            valsum = 1
        valnorm = valnorm / valsum * 100.0

        # show top-N features
        for i, (ch, hz) in enumerate(zip(chlist, hzlist)):
            if i >= cfg.FEAT_TOPN:
                break
            txt = '%-3s %5.1f Hz  normalized importance %-6s  raw importance %-6s  feature %-5d' %\
                  (ch, hz, '%.2f%%' % valnorm[i], '%.2f%%' % (values[i] * 100.0), keys[i])
            logger.info(txt)

        if cfg.EXPORT_GOOD_FEATURES:
            gfout.write('Importance(%) Channel Frequency Index\n')
            for i, (ch, hz) in enumerate(zip(chlist, hzlist)):
                gfout.write('%.3f\t%s\t%s\t%d\n' % (values[i]*100.0, ch, hz, keys[i]))
            gfout.close()


# for batch scripts
def batch_run(cfg_module):
    cfg = pu.load_config(cfg_module)
    cfg = check_config(cfg)
    run(cfg, interactive=True)

def run(cfg, state=mp.Value('i', 1), queue=None, interactive=False, cv_file=None, feat_file=None, logger=logger):

    redirect_stdout_to_queue(logger, queue, 'INFO')

    # add tdef object
    cfg.tdef = trigger_def(cfg.TRIGGER_FILE)

    # Extract features
    if not state.value:
        sys.exit(-1)
    featdata = features.compute_features(cfg)

    # Find optimal threshold for TPR balancing
    #balance_tpr(cfg, featdata)

    # Perform cross validation
    if not state.value:
        sys.exit(-1)

    if cfg.CV_PERFORM[cfg.CV_PERFORM['selected']] is not None:
        cross_validate(cfg, featdata, cv_file=cv_file)

    # Train a decoder
    if not state.value:
        sys.exit(-1)

    if cfg.EXPORT_CLS is True:
        train_decoder(cfg, featdata, feat_file=feat_file)

    with state.get_lock():
        state.value = 0



if __name__ == '__main__':
    # Load parameters
    if len(sys.argv) < 2:
        cfg_module = input('Config module name? ')
    else:
        cfg_module = sys.argv[1]
    batch_run(cfg_module)

    logger.info('Finished.')
