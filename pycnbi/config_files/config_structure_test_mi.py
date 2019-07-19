########################################################################
class Basic:
    """
    Contains the basic parameters for the online modality of Motor Imagery protocol
    """  
    #-------------------------------------------
    # Decoder
    #-------------------------------------------
    params1 = dict()
    params1.update({'DECODER_FILE': str})
    params1.update({'DIRECTIONS': ('L', 'R', 'U', 'D', 'B')})
    params1.update({'FAKE_CLS': (None, True)})

    #-------------------------------------------
    # feedback type
    #-------------------------------------------
    params2 = dict()
    params2.update({'FEEDBACK_TYPE': ('BAR', 'BODY')})
    params2.update({'FEEDBACK_IMAGE_PATH': str})
    params2.update({'REFRESH_RATE': int}) # Maximum refresh rate in Hz

    #-------------------------------------------
    # Trials
    #-------------------------------------------
    params3 = dict()
    params3.update({'TRIALS_EACH': int})
    params3.update({'TRIALS_RANDOMIZE': (False, True)})
    params3.update({'TRIALS_RETRY': (False, True)})
    params3.update({'TRIALS_PAUSE': (False, True)})

    #-------------------------------------------
    # Bar behavior
    #-------------------------------------------
    params3 = dict()
    params3.update({'PROB_ALPHA_NEW': float})
    params3.update({'BAR_BIAS': None})
    params3.update({'BAR_STEP': dict(left=int, right=int, up=int, down=int, both=int)})
    params3.update({'BAR_SLOW_START': {'False':None, 'True':list}})             # BAR_SLOW_START: None or in seconds
    params3.update({'BAR_REACH_FINISH': (False, True)})
    params3.update({'POSTIVE_FEEDBACK': (False, True)})

    #-------------------------------------------
    # Screen property
    #-------------------------------------------
    params4 = dict()
    params4.update({'SCREEN_SIZE': ((1920, 1080), (1600, 1200), (1680, 1050), (1280, 1024), (1024, 768))})
    params4.update({'SCREEN_POS': ((0, 0), (1920, 0))}) # TO CHANGE: add a check box second monitor and auto display on it. 


########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """
   
    #-------------------------------------------
    # Trigger device type
    #-------------------------------------------
    params1 = dict()
    params1.update({'TRIGGER_DEVICE': (None, 'ARDUINO','USB2LPT','SOFTWARE','DESKTOP')})
    params1.update({'TRIGGER_FILE': str}) # full list: PYCNBI_ROOT/Triggers/triggerdef_*.py

    #-------------------------------------------
    # acquisition device (set both to None to search)
    #-------------------------------------------
    params2 = dict()
    params2.update({'AMP_NAME': str})
    params2.update({'AMP_SERIAL': str})

    #-------------------------------------------
    # Timings
    #-------------------------------------------
    params3 = dict()

    params3.update({'TIMINGS':dict(INIT=float, GAP=float, READY=float, FEEDBACK=float, DIR_CUE=float, CLASSIFY=float)})

    params4 = dict()    
    params4.update({'SHOW_CUE': (False, True)})
    params4.update({'SHOW_RESULT': (False, True)})   # show the classification result
    params4.update({'SHOW_TRIALS': (False, True)})

    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params5 = dict()
    params5.update({'GLASS_USE': (False, True)})

    #-------------------------------------------
    # Debug
    #-------------------------------------------
    params5.update({'DEBUG_PROBS': (False, True)})    
    params5.update({'LOG_PROBS': (False, True)})

    #-------------------------------------------
    # Parallel decoding
    #-------------------------------------------
    params5.update({'PARALLEL_DECODING': {'False':None, 'True':{'period': float, 'num_strides': int}}})



    