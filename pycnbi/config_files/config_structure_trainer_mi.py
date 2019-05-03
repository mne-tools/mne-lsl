########################################################################
class Basic:
    """
    Contains the basic parameters for the training modality of Motor Imagery protocol
    """

    #-------------------------------------------
    # Data 
    #-------------------------------------------
    params1 = dict()
    params1.update({'DATADIR': None})                               # read all data files from this directory for training
    
    #-------------------------------------------
    # Events
    #-------------------------------------------
    params2 = dict()
    # Issue: Think about it, maybe split in 2
    params2.update({'EPOCH': [None, None]})                         # epoch ranges in seconds relative to onset
 
    #-------------------------------------------
    # Channels specification
    #-------------------------------------------
    params3 = dict()
    params3.update({'CHANNEL_PICKS': None})                         # Pick a subset of channels for PSD.
    params3.update({'EXCLUDES': None})                              # Overwrite the CHANNEL_PICKS
    params3.update({'REF_CH_OLD': None})                            # Recover this channel which was used as reference channel.
    params3.update({'REF_CH_NEW': None})                            # Re-reference to this set of channels, averaged if more than 1.

    #-------------------------------------------
    # Filters
    #-------------------------------------------
    params5 = dict()
    params5.update({'SP_FILTER': [None, 'car', 'laplacian']})       # apply spatial filter immediately after loading data
    params5.update({'SP_CHANNELS': None})                           # only consider the following channels while computing
    # apply spectrial filter immediately after applying SP_FILTER
    # Can be either overlap-add FIR or forward-backward IIR via filtfilt
        # if lfreq < hfreq: bandpass
        # if lfreq > hfreq: bandstop
        # if lfreq == None: highpass
        # if hfreq == None: lowpass
    params5.update({'TP_FILTER': None})                            # None or [lfreq, hfreq]
    params5.update({'NOTCH_FILTER': None})                         # None or list of values

    #-------------------------------------------
    # Parallel processing
    #-------------------------------------------
    params6 = dict()
    params6.update({'N_JOBS': None})                                 # number of cores used for parallel processing (set None to use all cores)


########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """

    #-------------------------------------------
    # Trigger
    #-------------------------------------------
    params1 = dict()                            
    params1.update({'TRIGGER_FILE': None})                          # which trigger file template
    params1.update({'TRIGGER_DEF': None})                         # which trigger set?
    
   
    #-------------------------------------------
    # Unit conversion
    #-------------------------------------------
    params2 = dict()
    params2.update({'MULTIPLIER': None})

    #-------------------------------------------
    # Feature types
    #-------------------------------------------
    params3 = dict()
    params3.update({'FEATURES': ['PSD', 'CSP', 'TIMELAG']})
    params3.update({'EXPORT_GOOD_FEATURES': [False, True]})
    params3.update({'FEAT_TOPN': None})                             # show only the top N features

    #-------------------------------------------
    # PSD 
    #-------------------------------------------
    # wlen: window length in seconds
    # wstep: window step in absolute samples (32 is enough for 512 Hz, or 256 for 2KHz)
    params5 = dict()
    params5.update({'PSD': dict(fmin=None, fmax=None, wlen=None, wstep=None)})

    #-------------------------------------------
    # Classifier
    #-------------------------------------------
    params6 = dict()
    params6.update({'CLASSIFIER': ['GB', 'XGB', 'RF', 'rLDA', 'LDA']})
    params6.update({'EXPORT_CLS': [False, True]})
    params6.update({'GB': dict(trees=1000, learning_rate=0.01, max_depth=3, seed=666)})
    params6.update({'RF': dict(trees=1000, max_depth=5, seed=666)})
    params6.update({'RLDA_REGULARIZE_COEFF': None})

    #-------------------------------------------
    # Cross-Validation & testing
    #-------------------------------------------
    params7 = dict()
    params7.update({'CV_PERFORM': [None, 'StratifiedShuffleSplit', 'LeaveOneOut']})
    params7.update({'CV_TEST_RATIO': None})                         # StratifiedShuffleSplit only
    params7.update({'CV_FOLDS': None })                             # StratifiedShuffleSplit only
    params7.update({'CV_RANDOM_SEED': None})                        # StratifiedShuffleSplit only
    params7.update({'CV_EXPORT_RESULT': [False, True]})             # Common