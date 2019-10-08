import pycnbi
import pycnbi.utils.q_common as qc
from epochs2psd import epochs2psd

# parameters
data_dir = r'D:\data\MI\rx1\offline\gait-pulling\20161104\test'
channel_picks = None
tmin = 0.0
tmax = 3.0
fmin = 1
fmax = 40
w_len = 0.5
w_step = 16
from pycnbi.triggers.trigger_def import trigger_def

tdef = trigger_def('triggerdef_16.ini')
events = {'left':tdef.LEFT_GO, 'right':tdef.RIGHT_GO}

if __name__ == '__main__':
    for f in qc.get_file_list(data_dir):
        if f[-4:] != '.fif': continue
        print(f)
        epochs2psd(f, channel_picks, events, tmin, tmax, fmin, fmax, w_len, w_step)
