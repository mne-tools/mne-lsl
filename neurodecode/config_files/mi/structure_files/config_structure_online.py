########################################################################
class Basic:
    """
    Contains the basic parameters for the online modality of Motor Imagery protocol
    """
    #-------------------------------------------
    # PATH TO SAVE DATA
    #-------------------------------------------
    params0 = dict()
    params0.update({'DATA_PATH': str})                              # Path to save the recording    

    #-------------------------------------------
    # Decoder
    #-------------------------------------------
    params1 = dict()
    params1.update({'DECODER_FILE': str})                           # Path to the saved decoder
    params1.update({'DIRECTIONS': ('L', 'R', 'U', 'D', 'B')})       # The feedback directions, correspond to the feedback squares (L:Left, R:Right, U: Up, D:Down, B:Both)
    params1.update({'FAKE_CLS': (None, True)})                      #  If true, use a random classifier

    #-------------------------------------------
    # feedback type
    #-------------------------------------------
    params2 = dict()
    params2.update({'FEEDBACK_TYPE': ('BAR', 'BODY')})              # The feedback type : BAR: classical bar, BODY: images (need to be provided), originaly used for body images 
    params2.update({'FEEDBACK_IMAGE_PATH': str})                    # If BODY feedback selected: images path
    params2.update({'REFRESH_RATE': int})                           # Feedback/decoder refresh rate in Hz

    #-------------------------------------------
    # Trials
    #-------------------------------------------
    params3 = dict()
    params3.update({'TRIALS_EACH': int})                            # Trial numbers of each class
    params3.update({'TRIALS_RANDOMIZE': (False, True)})             # Randomize trials order
    params3.update({'TRIALS_RETRY': (False, True)})                 # If missed trial, can retry
    params3.update({'TRIALS_PAUSE': (False, True)})                 # Pause between each trial

    #-------------------------------------------
    # Bar behavior
    #-------------------------------------------
    params4 = dict()
    params4.update({'PROB_ALPHA_NEW': float})                       # Probability exponential smoothing
    params4.update({'BAR_BIAS': None})                              # To compensate for a classifier biais toward one class
    params4.update({'BAR_STEP': dict(left=int, right=int, up=int, down=int, both=int)}) # The feedback displacement unit, influences the feedback speed
    params4.update({'BAR_SLOW_START': {'False':None, 'True':list}}) # Wait before starting decoding (secs)
    params4.update({'BAR_REACH_FINISH': (False, True)})             # If True, when feedback reaches 100, move to next trials. If false, wait until trial time is finished
    params4.update({'POSTIVE_FEEDBACK': (False, True)})             # Display only positive feedback increase, the feedback cannot go backward

    #-------------------------------------------
    # Screen property
    #-------------------------------------------
    params5 = dict()
    params5.update({'SCREEN_SIZE': ((1920, 1080), (1600, 1200), (1680, 1050), (1280, 1024), (1024, 768))})  # The monitor display
    params5.update({'SCREEN_POS': ((0, 0), (1920, 0))})             # For second monitor usage, assume it is on the right 


########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """
   
    #-------------------------------------------
    # Trigger device type
    #-------------------------------------------
    params1 = dict()
    params1.update({'TRIGGER_DEVICE': (None, 'ARDUINO','USB2LPT','SOFTWARE','DESKTOP')})        # The supported trigger type (cf. trigger module for more info)
    params1.update({'TRIGGER_FILE': str})                                                       # .ini file containing the triggers events mapping int-str (cf. config_files/mi/triggerdef.ini)

    #-------------------------------------------
    # acquisition device (set to None to search)
    #-------------------------------------------
    params2 = dict()
    params2.update({'AMP_NAME': str})                                                           # The LSL stream name to connect to

    #-------------------------------------------
    # Timings
    #-------------------------------------------
    params3 = dict()

    params3.update({'TIMINGS':dict(INIT=float, GAP=float, READY=float, FEEDBACK=float, DIR_CUE=float, CLASSIFY=float)})
    # The protocol timings
    # INIT: New trial, black screen
    # GAP: Display trial number (if TRIAL_PAUSE: pause here)
    # READY: Show empty feedback (fixate)
    # DIR_CUE: Cue display (one direction colored)
    # CLASSIFY : MI trial length
    # FEEDBACK: color the selected direction, if feedback reached 100 or trial time consumed

    params4 = dict()    
    params4.update({'SHOW_CUE': (False, True)})         # Show cue to the subject
    params4.update({'SHOW_RESULT': (False, True)})      # Show the classification accuracy (trial based) + confusion matrix
    params4.update({'SHOW_TRIALS': (False, True)})      # Show the trial number

    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params5 = dict()
    params5.update({'GLASS_USE': (False, True)})        # For google glass usage

    #-------------------------------------------
    # Debug
    #-------------------------------------------
    params5.update({'DEBUG_PROBS': (False, True)})      # Output and log (LOG_PROBS is True) all smoothed probabilities
    params5.update({'LOG_PROBS': (False, True)})        # Log to file all smoothed probabilities

    #-------------------------------------------
    # Parallel decoding
    #-------------------------------------------
    params5.update({'PARALLEL_DECODING': {'False':None, 'True':{'period': float, 'num_strides': int}}})     # If True, run decoding on multicores. 
    # If the decoder runs at 32ms per cycle, we can set period=0.04, stride=0.01, num_strides=4 to achieve 100 Hz decoding.



    