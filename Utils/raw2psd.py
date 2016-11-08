from __future__ import print_function, division

"""
Compute PSD features over a sliding window on the entire raw file

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

# start
import pycnbi_config
import pycnbi_utils as pu
import numpy as np
import q_common as qc
import mne
from IPython import embed # for debugging

def raw2psd(rawfile, fmin=1, fmax=40, wlen=0.5, wstep=1, tmin=0.0, tmax=None, channel_picks=None, excludes=None, n_jobs=1):
	"""
	Compute PSD features over a sliding window on the entire raw file

	Input
	=====
	rawfile: fif-format raw file
	channel_picks: None or list of channel indices
	tmin (sec): start time of the PSD window relative to the event onset.
	tmax (sec): end time of the PSD window relative to the event onset. None = until the end.
	fmin (Hz): minimum PSD frequency
	fmax (Hz): maximum PSD frequency
	wlen (sec): sliding window length for computing PSD (sec)
	wstep (int): sliding window step (time samples)
	excludes (list): list of channels to exclude
	"""

	raw, eve= pu.load_raw(rawfile)
	sfreq= raw.info['sfreq']
	wframes= int( round( sfreq * wlen ) )
	raw_eeg= raw.pick_types(meg=False, eeg=True, stim=False, exclude=excludes)
	if channel_picks is None:
		rawdata= raw_eeg._data
	else:
		# because indexing is messed up if excludes is not None
		raise (RuntimeError, 'channel_picks not supported yet')
		rawdata= raw_eeg._data[channel_picks]
	if tmax is None:
		tmax= rawdata.shape[1]
	else:
		tmax= int( round(tmax * sfreq) )

	psde= mne.decoding.PSDEstimator(sfreq, fmin=fmin, fmax=fmax, n_jobs=1, adaptive=False)
	psd_all= None
	evelist= []
	times= []
	last_eve= 0
	y_i= 0
	for t in range( wframes, tmax, wstep ):
		if t >= eve[y_i][0]:
			last_eve= eve[y_i][2]
			y_i += 1
		evelist.append(last_eve)
		window= rawdata[:, t-wframes:t]
		psd= psde.transform( window.reshape( (1, window.shape[0], window.shape[1]) ) )
		if psd_all is None:
			psd_all= psd
		else:
			psd_all= np.concatenate( (psd_all, psd) )
		times.append(t)

	[basedir, fname, fext]= qc.parse_path(rawfile)
	fout= '%s/psd-%s.pkl'% (basedir, fname)
	dataout= {'psd':psd_all, 'times':np.array(times), 'sfreq':sfreq,
		'channels':raw_eeg.ch_names, 'wframes':wframes, 'events':evelist}
	qc.save_obj(fout, dataout )
	print('Exported to %s'% fout)
