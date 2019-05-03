########################################################################
class Basic:
    """
    Contains the basic parameters for the online modality of Motor Imagery protocol
    """  

    # Data
    params1 = dict()
    params1.update({'DECODER_FILE': None})
    params1.update({'DIRECTIONS': [ ('L', 'LEFT_GO'), ('R', 'RIGHT_GO') ]})

    # feedback type
    params2 = dict()
    params2.update({'FEEDBACK_TYPE': ['BAR', 'BODY']})
    params2.update({'FEEDBACK_IMAGE_PATH': None})
    params2.update({'REFRESH_RATE': None}) # Maximum refresh rate in Hz

     # Trials
    params3 = dict()
    params3.update({'TRIALS_EACH': None})
    params3.update({'TRIALS_RANDOMIZE': [False, True]})
    params3.update({'TRIALS_RETRY': [False, True]})
 
    # Screen property
    params4 = dict()
    params4.update({'SCREEN_SIZE': [[1920, 1080], [1600, 1200], [1680, 1050], [1280, 1024], [1024, 768]]})
    params4.update({'SCREEN_POS': [[1920, 0], [1920, 1080]]}) # TO CHANGE: add a check box second monitor and auto display on it. 


########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """
   
    # Trigger device type
    params1 = dict()
    params1.update({'TRIGGER_DEVICE': [None, 'ARDUINO','USB2LPT','SOFTWARE','DESKTOP']})
    params1.update({'TRIGGER_FILE': None}) # full list: PYCNBI_ROOT/Triggers/triggerdef_*.py

    # acquisition device (set both to None to search)
    params2 = dict()
    params2.update({'AMP_NAME': None})
    params2.update({'AMP_SERIAL': None})

    # Timings
    params3 = dict()
    params3.update({'T_INIT': None})                # initial waiting time
    params3.update({'T_GAP': None})                 # intertrial gap
    params3.update({'T_READY': None})               # no direction, only dot cue
    params3.update({'T_FEEDBACK': None})            # decision feedback shown
    params3.update({'T_DIR_CUE': None})             # direction cue shown
    params3.update({'T_CLASSIFY': None})            # imagery period
    params3.update({'SHOW_CUE': [False, True]})
    params3.update({'SHOW_RESULT': [False, True]})  # show the classification result
    params3.update({'SHOW_TRIALS': [False, True]})

    #