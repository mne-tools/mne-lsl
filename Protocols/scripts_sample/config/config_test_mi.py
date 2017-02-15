# MI classifier
CLS_MI = r'D:\data\MI\q5\2016\train\classifier\classifier-64bit.pcl'

# trigger device
TRIGGER_DEVICE = 'ARDUINO'  # None | 'ARDUINO' | 'USB2LPT' | 'DESKTOP' | 'FAKE'
TRIGGER_DEF = 'triggerdef_16'  # see full list: ROOT/Triggers/triggerdef_*.py

# define bar direction for each class label: (bar direction, class label)
DIRECTIONS = [('U', 'UP_GO'), ('B', 'BOTH_GO')]

# number of trials for each direction
TRIALS_EACH = 20

# acquisition device (set both to None to search)
AMP_NAME = None
AMP_SERIAL = None

# timings
T_INIT = 10  # initial waiting time before starting
T_GAP = 4  # intertrial gap
T_READY = 2  # only cross is shown without any direction cue
T_DIR_CUE = 2  # direction cue shown
T_CLASSIFY = 5  # imagery period
T_FEEDBACK = 1  # decision feedback shown
REFRESH_RATE = 30  # maximum refresh rate in Hz

# evidence accumulation parameter
PROB_ACC_ALPHA = 0.8  # p_new= p_old * alpha + p_new * (1-alpha)

# bar bias: None or (direction, probability)
BAR_BIAS = None  # ('U',0.01)

# bar speed: multiplied to raw likelihoods
BAR_STEP = 20

# finish the trial if bar reaches any end
BAR_REACH_FINISH = False

# positive feedback only?
POSITIVE_FEEDBACK = False

# screen property
SCREEN_SIZE = (1680, 1050)
SCREEN_POS = (1920, 0)

# fake decoder?
FAKE_CLS = None  # None or 'left' or 'right' or 'middle' or 'random'

# don't change for now
CLS_TYPE = 'MI'

# communicate with Rex module?
WITH_REX = False

# use Google Glass?
GLASS_USE = False

# debug likelihoods
DEBUG_PROBS = False
