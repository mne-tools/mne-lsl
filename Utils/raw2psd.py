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
import os, mne
from IPython import embed # for debugging

def raw2psd(rawfile, fmin=1, fmax=40, wlen=0.5, wstep=1, tmin=0.0, tmax=None, channel_picks=None, excludes=None, n_jobs=1):
	"""
	Compute PSD features over a sliding window on the entire raw file.
	Leading edge of the window is the time reference.

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
		t_end= rawdata.shape[1]
	else:
		t_end= int( round(tmax * sfreq) )
	t_start= int( round(tmin * sfreq) ) + wframes
	psde= mne.decoding.PSDEstimator(sfreq, fmin=fmin, fmax=fmax, n_jobs=1, adaptive=False)
	print('[PID %d] %s'% (os.getpid(), rawfile) )
	psd_all= []
	evelist= []
	times= []
	t_len= t_end - t_start
	last_perc= 0
	last_eve= 0
	y_i= 0
	t_last= t_start
	tm= qc.Timer()
	for t in range( t_start, t_end, wstep ):
		# compute PSD
		window= rawdata[:, t-wframes : t]
		psd= psde.transform( window.reshape( (1, window.shape[0], window.shape[1]) ) )
		psd= psd.reshape( psd.shape[1], psd.shape[2] )
		psd_all.append( psd )
		times.append(t)

		# matching events at the current window
		if y_i < eve.shape[0] and t >= eve[y_i][0]:
			last_eve= eve[y_i][2]
			y_i += 1
		evelist.append(last_eve)

		perc= 1000.0 * (t - t_start) / t_len
		if int(perc) > last_perc:
			last_perc= int(perc)
			est= (1000 - last_perc) * tm.sec()
			fps= (t - t_last) / tm.sec()
			print('[PID %d] %.1f%% (%.1f FPS, %ds left)'% (os.getpid(), last_perc/10.0, fps, est) )
			t_last= t
			tm.reset()

	try:
		psd_all= np.array( psd_all )
		[basedir, fname, fext]= qc.parse_path(rawfile)
		fout_header= '%s/psd-%s-header.pkl'% (basedir, fname)
		fout_psd= '%s/psd-%s-data.npy'% (basedir, fname)
		header= {'psdfile':fout_psd, 'times':np.array(times), 'sfreq':sfreq,
			'channels':raw_eeg.ch_names, 'wframes':wframes, 'events':evelist}
		print('Exporting to:\n%s\n%s'% (fout_header, fout_psd))
		qc.save_obj(fout_header, header )
		np.save( fout_psd, psd_all )
		print('Exported.')
	except:
		print('(%s) Unexpected error occurred while saving. Dropping you into a shell for recovery.'% os.path.basename(__file__))
		embed()
