# Path to save data
DATA_PATH = r''

# classifier type
DECODER_FILE = r''

# fake decoder?
FAKE_CLS = None
DIRECTIONS = [ ('L', 'LEFT_GO'), ('R', 'RIGHT_GO') ]

# trigger device
TRIGGER_DEVICE = None # None | 'ARDUINO' | 'USB2LPT' | 'DESKTOP' | 'SOFTWARE'
TRIGGER_FILE = r''

# trial properties
TRIALS_EACH = 10
TRIALS_RANDOMIZE = True
TRIALS_RETRY = False
TRIALS_PAUSE = False

TIMINGS = { 'INIT':2,           \
            'GAP': 2,           \
            'READY': 2,         \
            'FEEDBACK': 1,    \
            'DIR_CUE': 1,     \
            'CLASSIFY': 5}                                

SHOW_CUE = True
SHOW_RESULT = True          # show the classification result
SHOW_TRIALS = True

# feedback type can be 'BAR' or 'BODY'
FEEDBACK_TYPE = 'BAR' # BAR | BODY
IMAGE_PATH = ''

'''
Bar behavior
'''
PROB_ALPHA_NEW = 0.02           # p_smooth = p_old * (1-PROB_ALPHA_NEW) + p_new * PROB_ALPHA_NEW
BAR_BIAS = ('L', 0.0)          # BAR_BIAS: None or (dir, prob)

BAR_STEP = {'left':5, 'right':5, 'up':5, 'down':5, 'both':5}
BAR_SLOW_START = {'selected':'False', 'False':None, 'True':[1.0]}            # BAR_SLOW_START: None or in seconds
# finish the trial if bar reaches the end
BAR_REACH_FINISH = True

# positive feedback only?
POSITIVE_FEEDBACK = False

# screen property
SCREEN_SIZE = (1920, 1080)
SCREEN_POS = (0, 0)

# use Google Glass?
GLASS_USE = False

# debug likelihoods
DEBUG_PROBS = True
LOG_PROBS = True

# high frequency parallel decoding (None or dict)
#PARALLEL_DECODING = None
PARALLEL_DECODING = {'selected':'False', 'False':None, 'True':{'period':0.06, 'num_strides':3}}

# visualization refresh rate
REFRESH_RATE = 30 # maximum refresh rate in Hz
