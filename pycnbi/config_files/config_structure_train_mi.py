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
    params1.update({'DIR_RANDOM': (False, True)})
    
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
    params2.update({'TIMINGS':dict(INIT=float, GAP=float, CUE=float, READY=float, READY_RANDOMIZE=float, DIR=float, DIR_RANDOMIZE=float)})

    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params3 = dict()
    params3.update({'GLASS_USE': (False, True)})
    
