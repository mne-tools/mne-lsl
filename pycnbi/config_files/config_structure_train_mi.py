########################################################################
class Basic:
    """
    Contains the basic parameters for the offline modality of Motor Imagery protocol
    """  
    
    #-------------------------------------------
    # Bar directions 
    #-------------------------------------------
    params1 = dict()
    params1.update({'DIRECTIONS': (None, 'L', 'R', 'U', 'D', 'B')})
    params1.update({'DIR_RANDOMIZE': (False, True)})
    
    #-------------------------------------------
    # feedback type
    #-------------------------------------------
    params2 = dict()
    params2.update({'FEEDBACK_TYPE': ('BAR', 'BODY')})
    params2.update({'FEEDBACK_IMAGE_PATH': str})
    params2.update({'REFRESH_RATE': int})              # Maximum refresh rate in Hz

    #-------------------------------------------
    # Trials 
    #-------------------------------------------
    params3 = dict()
    params3.update({'TRIALS_EACH': int})               # Trial numbers for each class
    params3.update({'TRIAL_PAUSE': (False, True)})      # Pause after each trial
    
    #-------------------------------------------
    # Screen property
    #-------------------------------------------
    params4 = dict()
    params4.update({'SCREEN_SIZE': ((1920, 1080), (1600, 1200), (1680, 1050), (1280, 1024), (1024, 768))})
    params4.update({'SCREEN_POS': ((0, 0), (1920, 0))}) # TO CHANGE: add a check box second monitor and auto display on it. 
    

########################################################################
class Advanced:
    """
    Contains the advanced parameters for the offline modality of Motor Imagery protocol
    """

    #-------------------------------------------
    # Trigger
    #-------------------------------------------
    params1 = dict()
    params1.update({'TRIGGER_DEVICE': (None, 'ARDUINO','USB2LPT','SOFTWARE','DESKTOP')})
    params1.update({'TRIGGER_FILE': str})          # full list: PYCNBI_ROOT/Triggers/triggerdef_*.py

    #-------------------------------------------
    # Timings
    #-------------------------------------------
    params2 = dict()
    params2.update({'T_INIT': int})                # initial waiting time
    params2.update({'T_GAP': int})                 # show how many trials left
    params2.update({'T_CUE': int})                 # no direction, only dot cue
    params2.update({'T_DIR_READY': int})           # green bar
    params2.update({'T_DIR_READY_RANDOMIZE': int}) # seconds
    params2.update({'T_DIR': int})                 # blue bar
    params2.update({'T_DIR_RANDOMIZE': int})       # seconds

    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params3 = dict()
    params3.update({'GLASS_USE': (False, True)})
    
