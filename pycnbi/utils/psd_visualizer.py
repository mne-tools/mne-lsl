"""
Real-time PSD visualization script

TODO: make it a function or a class

Author:
Kyuhwa Lee

"""

amp_name = 'openvibeSignal'
amp_serial = None

# screen
screen_offset_x = 100
screen_offset_y = 100

# channels
channel_picks = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

# PSD
w_seconds = 0.5
fmin = 71
fmax = 200

# filters
spatial = None #'car'
spatial_ch = None
spectral = None
spectral_ch = None
notch = None
notch_ch = None
multiplier = 1


# code begins
from mne.decoding import PSDEstimator
from pycnbi.stream_receiver.stream_receiver import StreamReceiver
import pycnbi.utils.pycnbi_utils as pu
import numpy as np
import cv2
import mne
import os
mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper


def get_psd(sr, psde, picks):
    sr.acquire()
    w, ts = sr.get_window()  # w = times x channels
    w = w.T  # -> channels x times

    # apply filters. Important: maintain the original channel order at this point.
    w = pu.preprocess(w, sfreq=sfreq, spatial=spatial, spatial_ch=spatial_ch,
                      spectral=spectral, spectral_ch=spectral_ch, notch=notch,
                              notch_ch=notch_ch, multiplier=multiplier)

    # select the same channels used for training
    w = w[picks]

    # debug: show max - min
    # c=1; print( '### %d: %.1f - %.1f = %.1f'% ( picks[c], max(w[c]), min(w[c]), max(w[c])-min(w[c]) ) )

    # psde.transform = [channels x freqs]
    psd = psde.transform(w)

    return psd

if __name__ == '__main__':
    sr = StreamReceiver(window_size=w_seconds, amp_name=amp_name, amp_serial=amp_serial)
    sfreq = sr.sample_rate
    psde = PSDEstimator(sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=None, adaptive=False,
                                     low_bias=True, n_jobs=1, normalization='length', verbose=None)
    ch_names = sr.get_channel_names()
    fq_res = 1 / w_seconds
    hz_list = []
    f = fmin
    while f <= fmax:
        hz_list.append(f)
        f += fq_res
    picks = [ch_names.index(ch) for ch in channel_picks]
    psd = get_psd(sr, psde, picks).T # freq x ch
    assert len(hz_list) == psd.shape[0], (len(hz_list), psd.shape[0])
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("img", screen_offset_x, screen_offset_y)
    #cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    mul_x = 50
    mul_y = 20
    fq_width = 55
    ch_height = 30
    img_x = psd.shape[1] * mul_x
    img_y = psd.shape[0] * mul_y
    img = np.zeros((img_y + ch_height, img_x + fq_width), np.uint8)
    # channels
    for x in range(psd.shape[1]):
        cv2.putText(img, '%s' % ch_names[picks[x]], (x * mul_x + fq_width + 13, img_y + ch_height - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, [255,255,255], 1, cv2.LINE_AA)
    # frequencies
    for y in range(psd.shape[0]):
        cv2.putText(img, '%5.0f' % hz_list[y], (1, y * mul_y + 15), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, [255,255,255], 1, cv2.LINE_AA)
    cv2.putText(img, 'Hz/Ch', (1, img_y + ch_height - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255,255,255], 1, cv2.LINE_AA)

    key = 0
    while key != 27:
        psd = get_psd(sr, psde, picks)
        psd_r = np.true_divide(psd, np.mean(psd, axis=0))
        psd_r = psd_r - np.min(psd_r, axis=0)
        psd_r = np.true_divide(psd_r, np.max(psd_r, axis=0))
        psd_n = np.round(psd_r * 255) # freq x ch

        psd_img = cv2.resize(psd_n.T, (img_x, img_y), interpolation=0)
        img[:-ch_height, fq_width:] = psd_img
        #cv2.imshow("img", cv2.cvtColor(cv2.cvtColor(img, code=cv2.COLOR_GRAY2BGR), code=cv2.COLOR_BGR2HSV))
        cv2.imshow("img", img)
        key = cv2.waitKey(1)
