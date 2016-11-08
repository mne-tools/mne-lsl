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

def epochs2psd(rawfile, channel_picks, event_id, tmin, tmax, fmin, fmax, w_len, w_step):
	"""
	Compute PSD features over a sliding window in epochs
	
	Exported data is 4D: [epochs] x [times] x [channels] x [freqs]

	Input
	=====
	rawfile: fif-format raw file
	channel_picks: None or list of channel indices
	event_id: { label(str) : event_id(int) }
	tmin: start time of the PSD window relative to the event onset
	tmax: end time of the PSD window relative to the event onset
	fmin: minimum PSD frequency
	fmax: maximum PSD frequency
	w_len: sliding window length for computing PSD
	w_step: sliding window step in time samples
	export: file name to be saved. It can have .mat or .pkl extension.
	- pkl extension exports in pickled Python numpy format.
	- mat extension exports in MATLAB format.
	"""

	rawfile= rawfile.replace('\\','/')
	raw, events= pu.load_raw(rawfile)
	sfreq= raw.info['sfreq']

	if channel_picks == None:
		picks= mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	else:
		picks= channel_picks

	# Epoching
	epochs= mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=False, picks=picks, baseline=(tmin,tmax), preload=True, add_eeg_ref=False)

	# Compute psd vectors over a sliding window between tmin and tmax
	w_len= int(sfreq * w_len) # window length
	psde= mne.decoding.PSDEstimator(sfreq, fmin=fmin, fmax=fmax, n_jobs=cpu_count(), adaptive=False)
	epochmat= { e:epochs[e]._data for e in event_id }
	psdmat= {}
	for e in event_id:
		# psd = [epochs] x [windows] x [channels] x [freqs]
		psd, _= pu.get_psd(epochs[e], psde, w_len, w_step, flatten=False)
		psdmat[e]= psd
		#psdmat[e]= np.mean(psd, 3) # for freq-averaged
	
	data= dict(epochs=epochmat, psds=psdmat, tmin=tmin, tmax=tmax, sfreq=epochs.info['sfreq'],\
		fmin= fmin, fmax= fmax, w_step=w_step, w_len=w_len, labels=epochs.event_id.keys())

	# Export
	[basedir, fname, fext]= qc.parse_path(rawfile)
	matfile= '%s/psd-%s.mat'% (basedir, fname)
	pklfile= '%s/psd-%s.pkl'% (basedir, fname)
	scipy.io.savemat( matfile, data )
	qc.save_obj( pklfile, data )
	print('Exported to %s'% matfile)
	print('Exported to %s'% pklfile)

