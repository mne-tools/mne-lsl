# -*- coding: utf-8 -*-
"""
Settings for trainer.py

Kyuhwa Lee, 2015
"""

from pycnbi.pycnbi_config import CAP
from pycnbi.triggers.trigger_def import trigger_def


########################################################################
class Basic:
    """
    Contains the basic parameters for the training modality of Motor Imagery protocol
    """ 

    '''"""""""""""""""""""""""""""
        DATA
    """""""""""""""""""""""""""'''
    # read all data files from this directory for training
    DATADIR = r'C:\LSL\pycnbi_local\z2\records\fif'

    # which trigger set?
    tdef = trigger_def('triggerdef_16.ini')
    TRIGGER_DEF = {tdef.LEFT_GO, tdef.RIGHT_GO}
    
    # epoch ranges in seconds relative to onset
    EPOCH = [0.5, 4.5]

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

    #CHANNEL_PICKS = None
    CHANNEL_PICKS= [ 'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'P3', 'Pz', 'P4' ] 
    #CHANNEL_PICKS = CAP['ANTNEURO_64_NO_PERIPHERAL']
    EXCLUDES = ['M1', 'M2', 'EOG']
    REF_CH = None
    #REF_CH = ['CPz', ['M1', 'M2']]

    '''"""""""""""""""""""""""""""
        FILTERS
    """""""""""""""""""""""""""'''
    # apply spatial filter immediately after loading data
    SP_FILTER = 'car' # None | 'car' | 'laplacian'
    # only consider the following channels while computing
    SP_CHANNELS = CHANNEL_PICKS # CHANNEL_PICKS # None | list

    # apply spectrial filter immediately after applying SP_FILTER
    # Can be either overlap-add FIR or forward-backward IIR via filtfilt
    # Value: None or [lfreq, hfreq]
    # if lfreq < hfreq: bandpass
    # if lfreq > hfreq: bandstop
    # if lfreq == None: highpass
    # if hfreq == None: lowpass
    #TP_FILTER = [0.6, 4.0]
    TP_FILTER = None

    NOTCH_FILTER = None # None or list of values
    
########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """

    '''"""""""""""""""""""""""""""
        EVENTS
    """""""""""""""""""""""""""'''
    # None or events filename (hardware events in raw file will be ignored)
    # TODO: set this flag in load_multi to concatenate frame numbers in multiple files.
    LOAD_EVENTS_FILE = None

    '''"""""""""""""""""""""""""""
    Parameters for computing PSD
    Ignored if LOAD_PSD == Ture

    wlen: window length in seconds
    wstep: window step in absolute samples (32 is enough for 512 Hz, or 256 for 2KHz)

    """""""""""""""""""""""""""'''
    LOAD_PSD = False

    # ignored if LOAD_PSD==True
    PSD= dict(fmin=1, fmax=40, wlen=0.5, wstep=32)


    '''"""""""""""""""""""""""""""
        UNIT CONVERSION
    """""""""""""""""""""""""""'''
    MULTIPLIER = 1 # V -> uV


    '''"""""""""""""""""""""""""""
        FEATURE TYPE
    """""""""""""""""""""""""""'''
    FEATURES = 'PSD' # PSD | CSP | TIMELAG
    EXPORT_GOOD_FEATURES = True
    FEAT_TOPN = 100 # show only the top N features

    # Wavelet parameters
    #DWT = dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 18])
    DWT = dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30])
    # export wavelets into MATLAB file
    EXPORT_DWT = False


    '''"""""""""""""""""""""""""""
        TimeLag parameters

    w_frames: window length in frames (samples) of downsampled data
    wstep: window step in downsampled data
    downsample: average every N-sample block (reduced to 1/N samples)
    """""""""""""""""""""""""""'''
    TIMELAG = dict(w_frames=10, wstep=5, downsample=100)


    '''"""""""""""""""""""""""""""
        CLASSIFIER
    """""""""""""""""""""""""""'''
    # clasifier
    CLASSIFIER = 'RF' # GB | XGB | RF | rLDA | LDA
    EXPORT_CLS = True

    # GB parameters
    GB = dict(trees=1000, learning_rate=0.01, max_depth=3, seed=666)

    # RF parameters
    RF = dict(trees=1000, max_depth=5, seed=666)

    # rLDA parameters
    RLDA_REGULARIZE_COEFF = 0.3

    '''"""""""""""""""""""""""""""
        CROSS-VALIDATION & TESTING
    """""""""""""""""""""""""""'''
    # 'StratifiedShuffleSplit' | 'LeaveOneOut' | None
    CV_PERFORM = 'StratifiedShuffleSplit' 
    CV_TEST_RATIO = 0.2 # ignored if LeaveOneOut
    CV_FOLDS = 8
    CV_RANDOM_SEED = 0
    CV_EXPORT_RESULT = True

    # testing file
    #ftest = 'C:/data/MI/q5/fif/test/20150310-212340.fif'
    ftest = ''


    '''"""""""""""""""""""""""""""
        ETC
    """""""""""""""""""""""""""'''
    # write to log file?
    USE_LOG = False

    # use CVA feature selection?
    USE_CVA = False

    # number of cores used for parallel processing (set None to use all cores)
    N_JOBS = 8
