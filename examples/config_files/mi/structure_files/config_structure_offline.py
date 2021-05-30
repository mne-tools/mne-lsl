########################################################################
class Basic:
    """
    Contains the basic parameters for the offline modality of Motor Imagery protocol
    """  

    #-------------------------------------------
    # PATH TO SAVE DATA
    #-------------------------------------------
    params0 = dict()
    params0.update({'DATA_PATH': str})                                  # Path to save the recorded data
    
    #-------------------------------------------
    # Bar directions 
    #-------------------------------------------
    params1 = dict()
    params1.update({'DIRECTIONS': (None, 'L', 'R', 'U', 'D', 'B')})     # The feedback directions, correspond to the feedback squares (L:Left, R:Right, U: Up, D:Down, B:Both)
    params1.update({'DIR_RANDOM': (False, True)})                       # Trials randomization
    
    #-------------------------------------------
    # feedback type
    #-------------------------------------------
    params2 = dict()
    params2.update({'FEEDBACK_TYPE': ('BAR', 'BODY')})                  # The feedback type : BAR: classical bar, BODY: images (need to be provided), originaly used for body images 
    params2.update({'FEEDBACK_IMAGE_PATH': str})                        # If BODY feedback selected: images path
    params2.update({'REFRESH_RATE': int})                               # Feedback refresh rate in Hz

    #-------------------------------------------
    # Trials 
    #-------------------------------------------
    params3 = dict()
    params3.update({'TRIALS_EACH': int})                                # Trial numbers of each class
    params3.update({'TRIAL_PAUSE': (False, True)})                      # Pause after each trial
    


########################################################################
class Advanced:
    """
    Contains the advanced parameters for the offline modality of Motor Imagery protocol
    """

    #-------------------------------------------
    # Trigger
    #-------------------------------------------
    params1 = dict()
    params1.update({'TRIGGER_DEVICE': (None, 'ARDUINO','USB2LPT','SOFTWARE','DESKTOP')})                # The supported trigger type (cf. trigger module for more info)
    params1.update({'TRIGGER_FILE': str})                                                               # .ini file containing the triggers events mapping int-str (cf. config_files/mi/triggerdef.ini)

    #-------------------------------------------
    # Timings
    #-------------------------------------------
    params2 = dict()
    params2.update({'TIMINGS':dict(INIT=float, GAP=float, CUE=float, READY=float, READY_RANDOMIZE=float, DIR=float, DIR_RANDOMIZE=float)})
                                                                                                         # The protocol timings
                                                                                                         # INIT: New trial, black screen
                                                                                                         # GAP: Display trial number (if TRIAL_PAUSE: pause here)
                                                                                                         # CUE: Show feedback empty (fixate)
                                                                                                         # READY: Cue display (one direction colored)
                                                                                                         # READY_RANDOMIZE: Add randomization timing range to READY
                                                                                                         # DIR: MI trial length
                                                                                                         # DIR_RANDOMIZE : Add randomization timing range to DIR

    #-------------------------------------------
    # Google Glass
    #-------------------------------------------
    params3 = dict()
    params3.update({'GLASS_USE': (False, True)})        # Google glass usage

    #-------------------------------------------
    # Screen property
    #-------------------------------------------
    params4 = dict()
    params4.update({'SCREEN_SIZE': ((1920, 1080), (1600, 1200), (1680, 1050), (1280, 1024), (1024, 768))})  # Monitor display
    params4.update({'SCREEN_POS': ((0, 0), (1920, 0))})                                                     # For second monitor usage, assume it is on the right 
    
