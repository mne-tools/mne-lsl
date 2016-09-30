# MI classifier
CLS_MI= r'D:\data\MI\q5\2016\train\classifier\classifier-64bit.pcl'

# trigger device
TRIGGER_DEVICE= 'ARDUINO' # None | 'ARDUINO' | 'USB2LPT' | 'DESKTOP' | 'FAKE'
TRIGGER_DEF= 'triggerdef_16' # see full list: ROOT/Triggers/triggerdef_*.py

# define bar direction for each class label: (bar direction, class label)
DIRECTIONS= [ ('U','UP_GO'), ('B','BOTH_GO') ]

# number of trials for each direction
TRIALS_EACH= 20

# acquisition device (set both to None to search)
AMP_NAME= None
AMP_SERIAL= None

# timings
T_INIT= 10 # initial waiting time before starting
T_GAP= 4 # intertrial gap
T_READY= 2 # only cross is shown without any direction cue
T_DIR_CUE= 2 # direction cue shown
T_CLASSIFY= 5 # imagery period

# evidence accumulation parameter
PROB_ACC_ALPHA= 0.8 # p_new= p_old * alpha + p_new * (1-alpha)

# bar bias: None or (direction, pixels)
BAR_BIAS= ('U',1)

# bar speed: multiplied to raw likelihoods
BAR_STEP= 10

# screen property
SCREEN_SIZE= (1680, 1050)
SCREEN_POS= (1920, 0)

# move only when correct?
POSITIVE_FEEDBACK= False

# finish the trial if the bar reached the end?
BAR_REACH_FINISH= False

# fake decoder?
FAKE_CLS= None # None or 'left' or 'right' or 'middle' or 'random'

# don't change for now
CLS_TYPE= 'MI' 

# communicate with Rex module?
WITH_REX= False

# use Google Glass?
GLASS_USE= False

# debug likelihoods
DEBUG_PROBS= False

# maximum refresh rate in Hz
REFRESH_RATE= 30
