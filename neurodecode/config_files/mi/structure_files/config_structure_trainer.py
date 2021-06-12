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
    params2.update({'EPOCH': list})                                         # Define the period of the MI task used to train the decoder (eg. [0.5, 3.5] = 0.5 sec after start to 3.5 sec).
 
    #-------------------------------------------
    # Channels specification
    #-------------------------------------------
    params3 = dict()
    params3.update({'PICKED_CHANNELS': list})                               # Pick a subset of channels for features computation (PSD).
    params3.update({'EXCLUDED_CHANNELS': list})                             # Overwrite the CHANNEL_PICKS
    params3.update({'REREFERENCE': {'False':None, \
                                    'True':dict(New=list, Old=list)}})      # For rereferencing (eg. dict(New:['M1','M2'], Old:['Cz'])), it recovers old ref

    #-------------------------------------------
    # Filters
    #-------------------------------------------
    params5 = dict()
    params5.update({'SP_FILTER': (None, 'car', 'laplacian')})               # Spatial filter: Common Average Reference or Laplacian
    params5.update({'SP_CHANNELS': list})                                   # only consider the following channels for spatial filtering
    
    params5.update({'TP_FILTER': {'False':None, 'True':list}})              # Temporal filter: None or [lfreq, hfreq]
    params5.update({'TP_CHANNELS': list})                                   # only consider the following channels for spatial filtering
    # Can be either overlap-add FIR or forward-backward IIR via filtfilt
        # if lfreq < hfreq: bandpass
        # if lfreq > hfreq: bandstop
        # if lfreq == None: highpass
        # if hfreq == None: lowpass
    params5.update({'NOTCH_FILTER': {'False':None, 'True':list}})           # Notch filter: None or list of values
    params5.update({'NOTCH_CHANNELS': list})                                # only consider the following channels for notch filtering

    #-------------------------------------------
    # Parallel processing
    #-------------------------------------------
    params6 = dict()
    params6.update({'N_JOBS': int})                                         # number of cores used for parallel processing (set None to use all cores)


########################################################################
class Advanced:
    """
    Contains the advanced parameters to train the Motor Imagery algorithm`
    """

    #-------------------------------------------
    # Trigger
    #-------------------------------------------
    params1 = dict()                            
    params1.update({'TRIGGER_FILE': str})                            # The trigger file containing the events mapping int-str 
    params1.update({'TRIGGER_DEF': str})                             # Which trigger set to train on (Define the number of classifier classes)
    params1.update({'LOAD_EVENTS': {'False':None, 'True':str}})      # Load events from .txt, timestamps need to be sample numbers 
    
   
    #-------------------------------------------
    # Unit conversion
    #-------------------------------------------
    params2 = dict()
    params2.update({'MULTIPLIER': int})                             # For unit conversion (eg. volts to uVolts)

    #-------------------------------------------
    # Feature types
    #-------------------------------------------
    params3 = dict()
    params3.update({'FEATURES': {'PSD':dict(fmin=int, fmax=int, wlen=float, wstep=int, decim=int)}})        # The features type: only Power Spectrum Density supported
    # fmin: The min frequency
    # fmax: The max frequency
    # wlen: The window length in seconds
    # wstep: The window step in absolute samples, used to simulate 
    # online decoding rate with a sliding window (32 is enough for 512 Hz, or 256 for 2KHz)    
    params3.update({'EXPORT_GOOD_FEATURES': (False, True)})                                                 # Export decoder good feature to good_features.txt
    params3.update({'FEAT_TOPN': int})                                                                      # show only the top N features on terminal

    #-------------------------------------------
    # Classifier
    #-------------------------------------------

    params6 = dict()
    # The classifier type: 
    # GB: Gradient Boosting, 
    # RF: Random Forest, 
    # rLDA: regularized LDA (implemented in NeuroDecode for two class classifier only)
    # LDA: Linear  Discriminant Analysis
    #  cf. scikit learn for more info
    params6.update({'CLASSIFIER':   {'GB': dict(trees=int, learning_rate=float, depth=int, seed=int), \
                                    'RF': dict(trees=int, depth=int, seed=int), \
                                    'rLDA': dict(r_coeff=float), \
                                    'LDA': dict(solver=('svd','lsqr','eigen'), shrinkage=(None, 'auto'))}})                                                        

    params6.update({'EXPORT_CLS': (False, True)})                                                           # Save the trained classifier to .pkl file
    params6.update({'SAVE_FEATURES': (False, True)})                                                        # Save the feature with classifier if online is adaptive 


    #-------------------------------------------
    # Cross-Validation & testing
    #-------------------------------------------
    params7 = dict()
    # Perform a k-fold Cross Validation before training the final classifier (used to predict online performance) Cf. scikit learn for more info
    params7.update({'CV_PERFORM':   {'False':None, \
                                    'StratifiedShuffleSplit': dict(test_ratio=float, folds=int, seed=int, export_result=(False, True)), \
                                    'LeaveOneOut': dict(export_result=(False, True))}})