"""
Settings for trainer.py

Kyuhwa Lee, 2015
"""

import pycnbi

'''"""""""""""""""""""""""""""
 EVENTS
"""""""""""""""""""""""""""'''
# Which trigger set?
from pycnbi.triggers.trigger_def import trigger_def

tdef = trigger_def('triggerdef_16.ini')
TRIGGER_DEF = {tdef.LEFT_GO, tdef.RIGHT_GO}

# None or external events filename (hardware events in raw file will be removed)
# TODO: set this flag in load_multi to concatenate frame numbers in multiple files.
LOAD_EVENTS_FILE = None

# epoch ranges in seconds relative to onset
EPOCH = [1.0, 4.5]

'''"""""""""""""""""""""""""""
 DATA
"""""""""""""""""""""""""""'''
# Reads all data files from this directory
DATADIR = r'D:\data\MI\rx1\train'

'''"""""""""""""""""""""""""""
 PSD PARAMETERS

 wlen: window length in seconds
 wstep: window step in frames (32 is enough for 512 Hz, or 256 for 2KHz)
"""""""""""""""""""""""""""'''
LOAD_PSD = False

# ignored if LOAD_PSD==True
PSD = dict(fmin=1, fmax=40, wlen=0.5, wstep=16)

'''"""""""""""""""""""""""""""
 CHANNEL SPECIFICATION

 CHANNEL_PICKS
 Pick a subset of channels for PSD. Note that Python uses zero-based indexing.
 However, for fif files saved using PyCNBI library, index 0 is the trigger channel
 and data channels start from index 1. (to be consistent with MATLAB)
 None: Use all channels. Ignored if LOAD_PSD == True

 REF_CH_NEW: Re-reference to this set of channels, averaged if more than 1.
 REF_CH_OLD: Recover this channel which was used as reference channel.
"""""""""""""""""""""""""""'''
CHANNEL_PICKS = ['Fz', 'FCz', 'Cz', 'FC1', 'FC2', 'FC3', 'FC4', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4']
EXCLUDES = ['M1', 'M2', 'EOG']
REF_CH_OLD = None  # 'CPz'
REF_CH_NEW = None  # ['M1','M2']

'''"""""""""""""""""""""""""""
 FILTERS
"""""""""""""""""""""""""""'''
# apply spatial filter immediately after loading data
SP_FILTER = 'car'  # None | 'car' | 'laplacian'
# only consider the following channels while computing
SP_CHANNELS = CHANNEL_PICKS  # CHANNEL_PICKS # None | list

# apply spectrial filter immediately after applying SP_FILTER
# Can be either overlap-add FIR or forward-backward IIR via filtfilt
# Value: None or [lfreq, hfreq]
# if lfreq < hfreq: bandpass
# if lfreq > hfreq: bandstop
# if lfreq is None: highpass
# if hfreq is None: lowpass
# TP_FILTER= [0.6, 4.0]
TP_FILTER = None

NOTCH_FILTER = None  # [50, 100, 150, 200, 250] # None or list of values

'''"""""""""""""""""""""""""""
 UNIT CONVERSION
"""""""""""""""""""""""""""'''
MULTIPLIER = 1  # default is Volts in MNE

'''"""""""""""""""""""""""""""
 FEATURE TYPE
"""""""""""""""""""""""""""'''
FEATURES = 'PSD'  # PSD | CSP | TIMELAG
EXPORT_GOOD_FEATURES = True  # Export informative features
FEAT_TOPN = 20  # export only the best-N features

'''"""""""""""""""""""""""""""
 CLASSIFIER
"""""""""""""""""""""""""""'''
# clasifier
CLASSIFIER = 'RF'  # GB | RF | rLDA | LDA
EXPORT_CLS = True

# GB parameters
GB = dict(trees=1000, learning_rate=0.01, max_depth=5, seed=666)

# RF parameters
RF = dict(trees=1000, max_depth=100, seed=666)

# rLDA parameters
RLDA_REGULARIZE_COEFF = 0.3

'''"""""""""""""""""""""""""""
 CROSS-VALIDATION & TESTING
"""""""""""""""""""""""""""'''
# do cross-validation?
CV_PERFORM = 'StratifiedShuffleSplit'  # 'StratifiedShuffleSplit' | 'LeaveOneOut' | None
CV_TEST_RATIO = 0.2  # ignored if LeaveOneOut
CV_FOLDS = 8
CV_RANDOM_SEED = 0
CV_EXPORT_RESULT = True

'''"""""""""""""""""""""""""""
 ETC
"""""""""""""""""""""""""""'''
# write to log file? (not used yet)
USE_LOG = False

# use pre-selected features using CVA?
USE_CVA = False

# number of cores used for parallel processing (set None to use all cores)
N_JOBS = None
