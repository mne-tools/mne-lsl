#-------------------------------------------
# Data 
#-------------------------------------------
DATA_PATH = ''

EPOCH = [0.5, 3.5]

#-------------------------------------------
# Trigger
#-------------------------------------------
TRIGGER_FILE = ''
TRIGGER_DEF = ['LEFT_GO', 'RIGHT_GO']
LOAD_EVENTS = {'selected':'False', 'False':None, 'True':r''}

#-------------------------------------------
# Channels specification
#-------------------------------------------
PICKED_CHANNELS= ['Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'P3', 'Pz', 'P4' ] 
EXCLUDED_CHANNELS = []

REREFERENCE = {'selected':'False', 'False':None, 'True':dict(New=['Cz'], Old=['M1/2', 'M2/2'])}

#-------------------------------------------
# Filters
#-------------------------------------------
SP_FILTER = 'car'
SP_CHANNELS = PICKED_CHANNELS
TP_FILTER = {'selected':'False', 'False':None, 'True':[1, 40]}
NOTCH_FILTER = {'selected':'False', 'False':None, 'True':[50]}

#-------------------------------------------
# Unit conversion
#-------------------------------------------
MULTIPLIER = 1

#-------------------------------------------
# Features
#-------------------------------------------
FEATURES = {'selected':'PSD','PSD':dict(fmin=1, fmax=40, wlen=0.5, wstep=10, decim=1)}
EXPORT_GOOD_FEATURES = True
FEAT_TOPN = 100

#-------------------------------------------
# Classifier
#-------------------------------------------

CLASSIFIER =    {'selected': 'RF', \
                'GB': dict(trees=1000, learning_rate=0.01, depth=3, seed=666), \
                'RF': dict(trees=1000, depth=5, seed=666), \
                'rLDA': dict(r_coeff=0.3), \
                'LDA': dict()}

EXPORT_CLS = True

#-------------------------------------------
# Cross-Validation & testing
#-------------------------------------------
CV_PERFORM =   {'selected':'StratifiedShuffleSplit', \
                'False':None, \
                'StratifiedShuffleSplit': dict(test_ratio=0.2, folds=8, seed=0, export_result=True), \
                'LeaveOneOut': dict(export_result=False)}

#-------------------------------------------
# Parallel processing
#-------------------------------------------
N_JOBS = 8
