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
    params3.update({'TRIALS_RANDOMIZE': [False, True]})
    params3.update({'TRIALS_RETRY': [False, True]})

    #-------------------------------------------
    # Bar behavior
    #-------------------------------------------
    params3 = dict()
    params3.update({'PROB_ALPHA_NEW': float})
    params3.update({'BAR_BIAS': None})
    params3.update({'BAR_STEP_LEFT': int})
    params3.update({'BAR_STEP_RIGHT': int})
    params3.update({'BAR_STEP_UP': int})
    params3.update({'BAR_STEP_DOWN': int})
    params3.update({'BAR_SLOW_START': int})             # BAR_SLOW_START: None or in seconds
    params3.update({'BAR_REACH_FINISH': (False, True)})
    params3.update({'POSTIVE_FEEDBACK': (False, True)})

    #-------------------------------------------
    # Screen property
    #-------------------------------------------
    params4 = dict()
    params4.update({'SCREEN_SIZE': ([1920, 1080], [1600, 1200], [1680, 1050], [1280, 1024], [1024, 768])})
    params4.update({'SCREEN_POS': ([1920, 0], [1920, 1080])})


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
    params3.update({'T_INIT': int})                  # initial waiting time
    params3.update({'T_GAP': int})                   # intertrial gap
    params3.update({'T_READY': int})                 # no direction, only dot cue
    params3.update({'T_FEEDBACK': int})              # decision feedback shown
    params3.update({'T_DIR_CUE': int})               # direction cue shown
    params3.update({'T_CLASSIFY': int})              # imagery period
    params3.update({'SHOW_CUE': (False, True)})
    params3.update({'SHOW_RESULT': (False, True)})   # show the classification result
    params3.update({'SHOW_TRIALS': (False, True)})


    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params4 = dict()
    params4.update({'GLASS_USE': (False, True)})

    #-------------------------------------------
    # Debug
    #-------------------------------------------
    params4.update({'DEBUG_PROBS': (False, True)})    
    params4.update({'LOG_PROBS': (False, True)})

    #-------------------------------------------
    # Parallel decoding
    #-------------------------------------------
    params4.update({'PARALLEL_DECODING': (None, {'True':{'period': float, 'num_strides': int}})})
    #params3.update({'PARALLEL_DECODING': (None, dict(period=0.06, num_strides=3))})



    