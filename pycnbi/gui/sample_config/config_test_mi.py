########################################################################
class Basic:
    """
    Contains the basic parameters for the Online modality of Motor Imagery protocol
    """  
    # classifier type
    CLS_MI = r'C:\LSL\pycnbi_local\z2\records\fif\classifier\classifier-64bit.pkl'
    DIRECTIONS = [ ('L', 'LEFT_GO'), ('R', 'RIGHT_GO') ]

    # trial properties
    TRIALS_EACH = 10
    TRIALS_RANDOMIZE = True
    TRIALS_RETRY = False

    # feedback type can be 'BAR' or 'BODY'
    FEEDBACK_TYPE = 'BAR' # BAR | BODY
    IMAGE_PATH = r'D:\work\pycnbi_protocols\BodyFeedback\BodyVisuals_behind.pkl'

    # p_smooth = p_old * (1-PROB_ALPHA_NEW) + p_new * PROB_ALPHA_NEW
    PROB_ALPHA_NEW = 0.02

    # Bar behavior
    # BAR_BIAS: None or (dir, prob)
    BAR_BIAS = None
    #BAR_BIAS = ('L', 0.05)
    # finish the trial if bar reaches the end
    BAR_REACH_FINISH = True
    # positive feedback only?
    POSITIVE_FEEDBACK = False

########################################################################
class Advanced:
    """
    Contains the advanced parameters for the Online modality of Motor Imagery protocol
    """


    # fake decoder?
    FAKE_CLS = None # None or 'left' or 'right' or 'middle' or 'random'
    
    # acquisition device (set both to None to search)
    #AMP_NAME = 'openvibeSignal'
    AMP_NAME = None
    AMP_SERIAL = None

    # trigger device
    TRIGGER_DEVICE = None # None | 'ARDUINO' | 'USB2LPT' | 'DESKTOP' | 'FAKE'
    TRIGGER_DEF = 'triggerdef_16' # see full list: ROOT/Triggers/triggerdef_*.py

    # timings
    T_INIT = 2 # initial waiting time before starting
    T_GAP = 1 # intertrial gap
    T_READY = 1 # only cross is shown without any direction cue
    T_FEEDBACK = 0.5 # decision feedback shown
    T_DIR_CUE = 1 # direction cue shown
    T_CLASSIFY = 5 # imagery period
    SHOW_CUE = True
    SHOW_RESULT = True # show the classification result
    SHOW_TRIALS = True
    '''
    FREE_STYLE = False
    if FREE_STYLE:
        T_INIT = 0 # initial waiting time before starting
        T_GAP = 0 # intertrial gap
        T_READY = 0 # only cross is shown without any direction cue
        T_FEEDBACK = 0.5 # decision feedback shown
        T_DIR_CUE = 0 # direction cue shown
        T_CLASSIFY = 666666 # imagery period
        SHOW_CUE = False
        SHOW_RESULT = True # show the classification result
        SHOW_TRIALS = False
    else:
        T_INIT = 5 # initial waiting time before starting
        T_GAP = 1 # intertrial gap
        T_READY = 2 # only cross is shown without any direction cue
        T_FEEDBACK = 0.5 # decision feedback shown
        T_DIR_CUE = 0.5 # direction cue shown
        T_CLASSIFY = 5 # imagery period
        SHOW_CUE = True
        SHOW_RESULT = True # show the classification result
        SHOW_TRIALS = True
    '''


    BAR_STEP_LEFT = 10
    BAR_STEP_RIGHT = 10
    BAR_STEP_UP = 10
    BAR_STEP_DOWN = 10
    # BAR_SLOW_START: None or in seconds
    BAR_SLOW_START = 1.0

    # screen property
    SCREEN_SIZE = (1920, 1080)
    #SCREEN_SIZE = (500, 500)
    SCREEN_POS = (0, 0)

    # don't change for now
    CLS_TYPE = 'MI' 

    # use Google Glass?
    GLASS_USE = False

    # debug likelihoods
    DEBUG_PROBS = True
    LOG_PROBS = True

    # high frequency parallel decoding (None or dict)
    PARALLEL_DECODING = None
    #PARALLEL_DECODING = {'period':0.06, 'num_strides':3}

    # visualization refresh rate
    REFRESH_RATE = 30 # maximum refresh rate in Hz
