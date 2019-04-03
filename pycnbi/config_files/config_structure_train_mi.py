########################################################################
class Basic:
    """
    Contains the basic parameters for the training modality of Motor Imagery protocol
    """  
    params = list()
    
    # feedback type
    params.append(['FEEDBACK_TYPE', ['BAR', 'BODY']])
    params.append(['IMAGE_PATH', None])

    # Bar directions 
    params.append(['DIRECTIONS', ['L', 'R', 'U', 'D', 'B']])
    
    # Randomized directions
    params.append(['DIR_RANDOMIZE', [True, False]])

    # Trials number
    params.append(['TRIALS_EACH'])

    # Timings
    params.append(['T_INIT']) # initial waiting time
    params.append(['T_GAP']) # show how many trials left
    params.append(['T_CUE']) # no direction, only dot cue
    params.append(['T_DIR_READY']) # green bar
    params.append(['T_DIR_READY_RANDOMIZE']) # seconds
    params.append(['T_DIR']) # blue bar
    params.append(['T_DIR_RANDOMIZE']) # seconds
    params.append(['T_RETURN']) # leg return time to neutral
    params.append(['T_STOP']) # stop after step

########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """
    params = list()

    # Trigger device type
    params.append(['TRIGGER_DEVICE', ['ARDUINO','USB2LPT','SOFTWARE','DESKTOP',None]])

    # full list: ROOT/Triggers/triggerdef_*.py
    # TO MODIFY: to a search GUI in the folder.
    params.append(['TRIGGER_DEF', ['triggerdef_gait_chuv']])


    # Screen property
    params.append(['SCREEN_SIZE', [[1920, 1080], [1600, 1200], [1680, 1050], [1280, 1024], [1024, 768]]])

    # Screen position (monitor 1 or 2)
    # TO CHANGE: add a check box second monitor and auto display on it. 
    params.append(['SCREEN_POS', [(1920, 0), (1920, 1080)]])




    # Pause after each trial
    params.append(['TRIAL_PAUSE', [True, False]])

    # Number of steps per trial
    params.append(['GAIT_STEPS'] 


    # Use Google Glass?
    params.append(['GLASS_USE', [True, False]])

    # Maximum refresh rate in Hz
    params.append(['REFRESH_RATE'])

    # STIMO project
    params.append(['WITH_STIMO', [True, False]])
    params.append(['STIMO_COMPORT'])
    params.append(['STIMO_BAUDRATE'])

