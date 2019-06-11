########################################################################
class Basic:
    """
    Contains the basic parameters to train the Motor Imagery algorithm
    """

    #-------------------------------------------
    # Data 
    #-------------------------------------------
    params1 = dict()
    params1.update({'DATA_PATH': str})                                      # read all data files from this directory to train on
    
    #-------------------------------------------
    # Events
    #-------------------------------------------
    params2 = dict()
    params2.update({'EPOCH': list})
 
    #-------------------------------------------
    # Channels specification
    #-------------------------------------------
    params3 = dict()
    params3.update({'PICKED_CHANNELS': list})                               # Pick a subset of channels for PSD.
    params3.update({'EXCLUDED_CHANNELS': list})                             # Overwrite the CHANNEL_PICKS
    params3.update({'REF_CHANNELS_OLD': list})                              # Recover this channel which was used as reference channel.
    params3.update({'REF_CHANNELS_NEW': list})                              # Re-reference to this set of channels, averaged if more than 1.

    #-------------------------------------------
    # Filters
    #-------------------------------------------
    params5 = dict()
    params5.update({'SP_FILTER': (None, 'car', 'laplacian')})               # apply spatial filter immediately after loading data
    params5.update({'SP_CHANNELS': list})                                   # only consider the following channels while computing
    # apply spectrial filter immediately after applying SP_FILTER
    # Can be either overlap-add FIR or forward-backward IIR via filtfilt
        # if lfreq < hfreq: bandpass
        # if lfreq > hfreq: bandstop
        # if lfreq == None: highpass
        # if hfreq == None: lowpass
    params5.update({'TP_FILTER': list})                                     # None or [lfreq, hfreq]
    params5.update({'NOTCH_FILTER': {'False':None, 'True':list}})         # None or list of values

    # example
    #params5.update({'TP_FILTER': {'None':[None, None], 'lowpass',[None, 30], 'bandpass':[1, 40], 'highpass':[8, None]}})


    #-------------------------------------------
    # Parallel processing
    #-------------------------------------------
    params6 = dict()
    params6.update({'N_JOBS': int})                               # number of cores used for parallel processing (set None to use all cores)


########################################################################
class Advanced:
    """
    Contains the advanced parameters to train the Motor Imagery algorithm`
    """

    #-------------------------------------------
    # Trigger
    #-------------------------------------------
    params1 = dict()                            
    params1.update({'TRIGGER_FILE': str})                            # which trigger file template
    params1.update({'TRIGGER_DEF': str})                             # which trigger set?
    
   
    #-------------------------------------------
    # Unit conversion
    #-------------------------------------------
    params2 = dict()
    params2.update({'MULTIPLIER': int})

    #-------------------------------------------
    # Feature types
    #-------------------------------------------
    params3 = dict()
     # wlen: window length in seconds
    # wstep: window step in absolute samples (32 is enough for 512 Hz, or 256 for 2KHz)
    params3.update({'FEATURES': {'PSD':dict(fmin=int, fmax=int, wlen=float, wstep=int)}})
    params3.update({'EXPORT_GOOD_FEATURES': (False, True)})
    params3.update({'FEAT_TOPN': int})                             # show only the top N features

    #-------------------------------------------
    # Classifier
    #-------------------------------------------
    #{'CLASSIFIER': {'GB':dict(trees=1000, learning_rate=0.01, max_depth=3, seed=666)}, 'RF':dict()}}

    params6 = dict()
    params6.update({'CLASSIFIER':   {'GB': dict(trees=int, learning_rate=float, max_depth=int, seed=int), \
                                    'RF': dict(trees=int, max_depth=int, seed=int), \
                                    'rLDA': dict(rlda_regularize_coeff=float), \
                                    'LDA': dict()}})

    params6.update({'EXPORT_CLS': (False, True)})


    #-------------------------------------------
    # Cross-Validation & testing
    #-------------------------------------------
    params7 = dict()
    params7.update({'CV_PERFORM':   {'False':None, \
                                    'StratifiedShuffleSplit': dict(CV_TEST_RATIO=float, CV_FOLDS=int, CV_RANDOM_SEED=int, CV_EXPORT_RESULT=tuple()), \
                                    'LeaveOneOut': dict(CV_EXPORT_RESULT=tuple())}})