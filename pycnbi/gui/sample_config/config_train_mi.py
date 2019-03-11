########################################################################
class Basic:
    """
    Contains the basic parameters for the training modality of Motor Imagery protocol
    """  
    # feedback type can be 'BAR' or 'BODY'
    FEEDBACK_TYPE = 'BAR'
    
    # path to the BODY feedback image
    IMAGE_PATH = None
    
    # Bar direction definition: L, R, U, D, B (both hands)
    DIRECTIONS = ['L', 'R']
    DIR_RANDOMIZE = True
    
    # Trials
    TRIALS_EACH = 10
    
    # Timings
    T_INIT = 5 # initial waiting time
    T_GAP = 2 # show how many trials left
    T_CUE = 1 # no bar, only red dot
    T_DIR_READY = 0.5 # green bar
    T_DIR = 5 # blue bar
    #T_RETURN = 1 # leg return time to neutral
    #T_STOP = 3 # stop after step
    
########################################################################
class Advanced:
    """
    Contains the advanced parameters for the training modality of Motor Imagery protocol
    """
    # Trigger device type ['ARDUINO','USB2LPT','SOFTWARE','DESKTOP',None]
    TRIGGER_DEVICE = 'SOFTWARE'
    
    # full list: PYCNBI_ROOT/Triggers/triggerdef_*.py
    TRIGGER_DEF = 'triggerdef_16'
    
    # Screen property
    SCREEN_SIZE = (1920, 1080)
    SCREEN_POS = (0, 0)
    
    # Trials
    # number of steps per trial
    GAIT_STEPS = 1
    # jitter for each step length
    RANDOMIZE_LENGTH = 0.0
    
    # Use Google Glass?
    GLASS_USE = False
    
    # Maximum refresh rate in Hz
    REFRESH_RATE = 30
    
    # STIMO project
    WITH_STIMO = False
    STIMO_COMPORT = 'COM6'
    STIMO_BAUDRATE = 9600
    
    
        
        
    
    