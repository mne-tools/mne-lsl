from __future__ import print_function, division

"""
Compute PSD features over a sliding window in epochs

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import pycnbi_config
import pycnbi_utils as pu
import scipy.io, mne
import numpy as np
import q_common as qc
from multiprocessing import cpu_count

DATAFILE= r'D:\data\MI\rx1\train\20160601-105104-raw.fif'
CHANNEL_PICKS= None # [5,9,11,12,13,14,15,16]
TMIN= 0.0
TMAX= 4.0
PSD= dict(fmin=1, fmax=40, w_len=1.0, w_step=16)
from triggerdef_16 import TriggerDef as tdef
EVENT_ID= {'left':tdef.LEFT_GO}

MATFILE= 'psd_windows.mat'
PKLFILE= 'psd_windows.pkl'


if __name__=='__main__':
	raw, events= pu.load_raw(DATAFILE)
	sfreq= raw.info['sfreq']

	if CHANNEL_PICKS == None:
		picks= mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	else:
		picks= CHANNEL_PICKS

	# Epoching
	epochs= mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX, proj=False, picks=picks, baseline=(TMIN,TMAX), preload=True, add_eeg_ref=False)

	# Compute psd vectors over a sliding window between TMIN and TMAX
	w_len= int(sfreq * PSD['w_len']) # window length
	psde= mne.decoding.PSDEstimator(sfreq, fmin=PSD['fmin'], fmax=PSD['fmax'], n_jobs=cpu_count(), adaptive=False)
	epochmat= { e:epochs[e]._data for e in EVENT_ID }
	psdmat= {}
	for e in EVENT_ID:
		# psd = [epochs] x [windows] x [channels] x [freqs]
		psd, _= pu.get_psd(epochs[e], psde, w_len, PSD['w_step'], flatten=False)
		psdmat[e]= psd
		#psdmat[e]= np.mean(psd, 3) # for freq-averaged
	
	data= dict(epochs=epochmat, psds=psdmat, tmin=TMIN, tmax=TMAX, sfreq=epochs.info['sfreq'],\
		wstep=PSD['w_step'], fmin= PSD['fmin'], fmax= PSD['fmax'], labels=epochs.event_id.keys())

	# Export
	[basedir, fname, fext]= qc.parse_path(DATAFILE)
	if MATFILE is not None:
		scipy.io.savemat('%s/%s'% (basedir, MATFILE), data )
		print('Exported to %s'% MATFILE)

	if PKLFILE is not None:
		qc.save_obj( '%s/%s'% (basedir, PKLFILE), psd )
		print('Exported to %s'% PKLFILE)
