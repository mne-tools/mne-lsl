import numpy as np
from pycnbi.triggers.trigger_def import trigger_def
tdef = trigger_def('triggerdef_16.ini')

'''"""""""""""""""""""""""""""
 DATA SOURCE
"""""""""""""""""""""""""""'''
DATA_DIRS = [r'C:\Data\Records\fif']
EXPORT_PATH = r'C:\Data\Records\fif\plots'
#CHANNEL_PICKS = None
CHANNEL_PICKS = ['CP3','CP4','C3','C4','CZ']


'''"""""""""""""""""""""""""""
 Reference channel
"""""""""""""""""""""""""""'''
# REREFERENCE: None | [CHANNEL_TO_RECOVER, [NEW_REF_CHANNELS]]
#REREFERENCE = ['CPz', ['M1', 'M2']]
REREFERENCE = None

'''"""""""""""""""""""""""""""
 Epochs and events of interest
"""""""""""""""""""""""""""'''
TRIGGERS = {tdef.LEFT_GO, tdef.RIGHT_GO}
EPOCH = [0.5, 4.5] # +2.0 relative to READY is the onset (GO), so -1 to +3s
EVENT_FILE = None

'''"""""""""""""""""""""""""""
 Baseline relative to onset while plotting

 BS_TIMES:
  None in index 0: beginning of data (epoch)
  None in index 1: end of data (epoch)
 BS_MODE:
  None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'

"""""""""""""""""""""""""""'''
BS_TIMES = (None, 1)
BS_MODE = 'logratio' 

'''"""""""""""""""""""""""""""
 TFR type: 'multitaper' or 'morlet'
"""""""""""""""""""""""""""'''
TFR_TYPE = 'multitaper'

'''"""""""""""""""""""""""""""
 PSD: freq range in numpy array
"""""""""""""""""""""""""""'''
FREQ_RANGE = np.arange(1, 40, 1)

'''"""""""""""""""""""""""""""
 Unit conversion
"""""""""""""""""""""""""""'''
MULTIPLIER = 1

'''"""""""""""""""""""""""""""
 Filters
"""""""""""""""""""""""""""'''
# apply spatial filter immediately after loading data
SP_FILTER = 'car' # None | 'car'
SP_CHANNELS = CHANNEL_PICKS # None | [ch1,ch2...]

# apply spectrial filter immediately after applying SP_FILTER
# Can be either overlap-add FIR or forward-backward IIR via filtfilt
# Value: None or [lfreq, hfreq]
# if lfreq < hfreq: bandpass
# if lfreq > hfreq: bandstop
# if lfreq == None: highpass
# if hfreq == None: lowpass
#TP_FILTER= [0.2, 210.0]
TP_FILTER = None

#NOTCH_FILTER= np.arange(50, 251, 50) # None or list of values
NOTCH_FILTER = None

'''"""""""""""""""""""""""""""
 Averaged power or power for each epoch?
"""""""""""""""""""""""""""'''
POWER_AVERAGED = True
VMIN = None
VMAX = None

'''"""""""""""""""""""""""""""
 Export image, MATLAB
"""""""""""""""""""""""""""'''
EXPORT_PNG = True
EXPORT_MATLAB = True

N_JOBS = 8
