import pycnbi_config
import numpy as np
import q_common as qc
from triggerdef_16 import TriggerDef
tdef= TriggerDef()

DATA_DIRS= [r'D:\data\MI\rx1\train']
CHANNEL_PICKS= [5,6,7,11]

'''"""""""""""""""""""""""""""
 Epochs and events of interest
"""""""""""""""""""""""""""'''
TRIGGERS= {tdef.LEFT_GO, tdef.RIGHT_GO}
EPOCH= [-2.0, 4.0]
EVENT_FILE= None

'''"""""""""""""""""""""""""""
 Baseline relative to onset while plotting
 None in index 0: beginning of data
 None in index 1: end of data
"""""""""""""""""""""""""""'''
BS_TIMES= (None,0)

'''"""""""""""""""""""""""""""
 PSD
"""""""""""""""""""""""""""'''
FREQ_RANGE= np.arange(1, 40, 1)

'''"""""""""""""""""""""""""""
 Unit conversion
"""""""""""""""""""""""""""'''
MULTIPLIER= 10**6 # (V->uV)

'''"""""""""""""""""""""""""""
 Filters
"""""""""""""""""""""""""""'''
# apply spatial filter immediately after loading data
SP_FILTER= 'car' # None | 'laplacian' | 'car'
SP_CHANNELS= CHANNEL_PICKS # None | dict (for Laplacian)

# apply spectrial filter immediately after applying SP_FILTER
# Can be either overlap-add FIR or forward-backward IIR via filtfilt
# Value: None or [lfreq, hfreq]
# if lfreq < hfreq: bandpass
# if lfreq > hfreq: bandstop
# if lfreq == None: highpass
# if hfreq == None: lowpass
#TP_FILTER= [0.6, 48.0]
TP_FILTER= None

NOTCH_FILTER= [50.0]

'''"""""""""""""""""""""""""""
 Averaged power or power for each epoch?
"""""""""""""""""""""""""""'''
POWER_AVERAGED= True

# Y-axis min, max
VMIN= -1
VMAX= 1
