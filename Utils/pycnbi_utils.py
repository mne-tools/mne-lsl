from __future__ import print_function, division

"""
PyCNBI utility functions

Note:
When exporting to Panda Dataframes format, raw.as_data_frame() silently
scales data to the Volts unit by default, which is the convention in MNE.
Try raw.as_data_frame(scalings=dict(eeg=1.0, misc=1.0))

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os, sys
import scipy.io
import mne
import numpy as np
import multiprocessing as mp
import q_common as qc
from pycnbi_config import CAP, LAPLACIAN
from scipy.signal import butter, lfilter,lfiltic, buttord
from IPython import embed

def slice_win(epochs_data, w_starts, w_length, psde, picks=None, epoch_id=None, flatten=True, verbose=False):
	'''
	Compute PSD values of a sliding window

	Params
		epochs_data: [channels] x [samples]
		w_starts: starting indices of sample segments
		w_length: window length in number of samples
		psde: MNE PSDEstimator object
		picks: subset of channels within epochs_data
		epochs_id: just to print out epoch ID associated with PID
		flatten: generate concatenated feature vectors
			If True: X = [windows] x [channels x freqs]
			If False: X = [windows] x [channels] x [freqs]

	Returns:
		[windows] x [channels*freqs] or [windows] x [channels] x [freqs]
	'''

	# raise error for wrong indexing
	def WrongIndexError(Exception):
		sys.stderr.write('\nERROR: %s\n'% Exception)
		sys.exit(-1)

	w_length= int(w_length)

	if epoch_id is None:
		print('[PID %d] Frames %d-%d'% (os.getpid(), w_starts[0], w_starts[-1] + w_length-1) )
	else:
		print('[PID %d] Epoch %d, Frames %d-%d'% (os.getpid(), epoch_id, w_starts[0], w_starts[-1] + w_length-1) )

	X= None
	for n in w_starts:
		n= int(n)
		if n >= epochs_data.shape[1]:
			raise WrongIndexError('w_starts has an out-of-bounds index %d for epoch length %d.'% (n,epochs_data.shape[1]))
		window= epochs_data[:, n:(n + w_length)]

		# dimension: psde.transform( [epochs x channels x times] )
		psd= psde.transform( window.reshape( (1, window.shape[0], window.shape[1]) ) )
		psd= psd.reshape( (psd.shape[0], psd.shape[1]*psd.shape[2]) )
		if picks:
			psd= psd[0][picks]
			psd= psd.reshape( (1, len(psd)) )

		if X is None:
			X= psd
		else:
			X= np.concatenate( (X, psd ), axis=0 )

		if verbose==True:
			print('[PID %d] processing frame %d / %d'% (os.getpid(), n, w_starts[-1]) )

	return X

def get_psd(epochs, psde, wlen, wstep, picks=None, flatten=True):
	"""
	Offline computation of multi-taper PSDs over a sliding window

	Params
	epochs: MNE Epochs object
	psde: MNE PSDEstimator object
	wlen: window length in frames
	wstep: window step in frames
	picks: channel picks
	flatten: boolean, see Returns section

	Returns
	-------
	if flatten==True:
		X_data: [epochs] x [windows] x [channels*freqs]
	else:
		X_data: [epochs] x [windows] x [channels] x [freqs]
	y_data: [epochs] x [windows]
	picks: feature indices to be used; use all if None

	TODO:
		Accept input as numpy array as well, in addition to Epochs object
	"""

	labels= epochs.events[:, -1]
	epochs_data= epochs.get_data()

	print('Opening pool of workers')
	pool= mp.Pool( mp.cpu_count() )

	# sliding window
	w_starts= np.arange(0, epochs_data.shape[2] - wlen, wstep)
	X_data= None
	y_data= None
	results= []
	for ep in np.arange(len(labels)):
		# for debugging (results not saved)
		#slice_win(epochs_data, w_starts, wlen, psde, picks, ep)

		# parallel psd computation
		results.append( pool.apply_async(slice_win, [epochs_data[ep], w_starts, wlen, psde, picks, ep]) )

	for ep in range(len(results)):
		r= results[ep].get() # windows x features
		X= r.reshape( (1, r.shape[0], r.shape[1]) ) # 1 x windows x features
		if X_data is None: X_data= X
		else: X_data= np.concatenate( (X_data, X), axis=0 )

		# speed comparison: http://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
		y= np.empty( (1, r.shape[0]) ) # 1 x windows
		y.fill( labels[ep] )
		if y_data is None: y_data= y
		else: y_data= np.concatenate( (y_data, y), axis=0 )
	pool.close()
	pool.join()

	if flatten:
		return X_data, y_data
	else:
		xs= X_data.shape
		nch= len(epochs.ch_names)
		return X_data.reshape(xs[0], xs[1], nch, int(xs[2]/nch)), y_data

# note that MNE already has find_events function
def find_events(events_raw):
	"""
	Find trigger values, rising from zero to non-zero
	"""
	events= [] # triggered event values other than zero

	# set epochs (frame start, frame end)
	ev_last= 0
	for et in range( len(events_raw) ):
		ev= events_raw[et]
		if ev != ev_last:
			if ev > 0:
				events.append( [ et, 0, ev ] )
			ev_last= ev

	return events

def find_event_channel(raw):
	"""
	Find event channel using heuristics for pcl files.

	Disclaimer: Not guaranteed to work.

	Input:
		raw: mne.io.RawArray object or numpy array (n_channels x n_samples)

	Output:
		channel index or None if not found.
	"""

	ech= None
	if type(raw) == np.ndarray:
		signals= raw
	else:
		signals= raw._data
		for ech_name in raw.ch_names:
			if 'TRIGGER' in ech_name  or 'STI ' in ech_name:
				# find a value changing from zero to a non-zero value
				ech= raw.ch_names.index(ech_name)
				break

	if ech is None:
		for c in range( signals.shape[0] ):
			if ( signals[c].astype(int)==signals[c] ).all() \
			and max(signals[c]) < 256 and min(signals[c])==0:
				ech= c
				break

	return ech

def raw2mat(infile, outfile):
	'''
	Convert raw data file to MATLAB file
	'''
	raw, events= load_raw(infile)
	header= dict(bads=raw.info['bads'], ch_names=raw.info['ch_names'],\
		sfreq=raw.info['sfreq'], events=events)
	scipy.io.savemat(outfile, dict(signals=raw._data, header=header) )
	print('\n>> Exported to %s'% outfile)

def pcl2mat_old(fpcl):
	"""
	For old format data only
	"""
	raw= qc.load_obj(fpcl)
	assert type(raw['signals'])==type(list())
	signals= np.array( raw['signals'][0] ) # samples x channels
	ts= raw['timestamps'][0]
	srate= raw['sample_rate']
	n_ch= raw['channels']
	if n_ch > 17: # BioSemi
		ev16= signals[:,0]-1 # first channel is event channel
		events_raw= 0xFF & ev16.astype(int) # keep only the low 8 bits
		events= find_events( events_raw )
	else:
		events= find_events( signals[:,-1] )

	print('Signal dimension:', signals.shape)
	print('Timestamp dimension:', len(ts) )
	print('Sampling rate:', srate)
	print('No. channels:', n_ch)
	data= dict(signals=signals, timestamps=ts, events=events, sample_rate=srate, n_channels=n_ch)
	fmat= fpcl[:-4] + '.mat'
	scipy.io.savemat( fmat, data )
	print('Saved data as', fmat)

def add_events_raw(rawfile, outfile, eventfile, overwrite=True):
	"""
	Add events from a file and save

	Note: If the event values already exists in raw file, the new event values
		will be added to the previous value instead of replacing them.
	"""

	raw= mne.io.Raw(rawfile, preload=True, proj=False)
	events= mne.read_events(eventfile)
	raw.add_events( events, stim_channel='TRIGGER' )
	raw.save( outfile, overwrite=overwrite )

def export_morlet(epochs, filename):
	"""
	Export wavelet tranformation decomposition into Matlab format
	"""
	freqs= np.array( DWT['freqs'] ) # define frequencies of interest
	n_cycles= freqs / 2.  # different number of cycle per frequency
	power, itc= mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
		n_cycles=n_cycles, use_fft=False, return_itc=True, n_jobs=mp.cpu_count() )
	scipy.io.savemat(filename, dict(power=power.data, itc=itc.data, freqs=freqs,\
		channels=epochs.ch_names, sfreq=epochs.info['sfreq'], onset=-epochs.tmin))

def event_timestamps_to_indices(sigfile, eventfile):
	"""
	Convert LSL timestamps to sample indices for separetely recorded events.

	Parameters:
	sigfile: raw signal file (Python Pickle) recorded with stream_recorder.py.
	eventfile: event file where events are indexed with LSL timestamps.

	Returns:
	events list, which can be used as an input to mne.io.RawArray.add_events().
	"""

	raw= qc.load_obj(sigfile)
	ts= raw['timestamps'].reshape(-1)
	ts_min= min(ts)
	ts_max= max(ts)
	events= []

	with open(eventfile) as f:
		for l in f:
			data= l.strip().split('\t')
			event_ts= float( data[0] )
			event_value= int( data[2] )
			# find the first index not smaller than ts
			next_index= np.searchsorted(ts, event_ts)
			if next_index >= len(ts):
				qc.print_c( '** WARNING: Event %d at time %.3f is out of time range (%.3f - %.3f).'% (event_value,event_ts,ts_min,ts_max), 'y' )
			else:
				events.append( [next_index, 0, event_value] )
			#print(events[-1])

	return events


def rereference(raw, ref_new, ref_old=None):
	"""
	Reference to new channels. raw object is modified in-place for efficiency.

	ref_new: None | list of str (RawArray) | list of int (numpy array)
		Reference to ref_new channels.

	ref_old: None | str
		Recover the original reference channel values. Only mne.io.RawArray is supported for now.
	"""

	# Re-reference and recover the original reference channel values if possible
	if type(raw) == np.ndarray:
		if raw_ch_old is not None:
			raise RuntimeError('Recovering original reference channel is not yet supported for numpy arrays.')
		assert type(raw_ch_new[0]) is int, 'Channels must be integer values for numpy arrays'
		raw -= np.mean( raw[ ref_new ], axis=0 )
	else:
		if ref_old is not None:
			# Add a blank (zero-valued) channel
			mne.io.add_reference_channels(raw, ref_old, copy=False)
		# Re-reference
		mne.io.set_eeg_reference(raw, ref_new, copy=False)

	return True


def preprocess(raw, sfreq=None, spatial=None, spatial_ch=None, spectral=None, spectral_ch=None,
	notch=None, notch_ch=None, multiplier=1, ch_names=None):
	"""
	Apply spatial, spectral, notch filters and convert unit.
	raw is modified in-place.

	Input
	------
	raw: mne.io.RawArray | mne.Epochs | numpy.array (n_channels x n_samples)

	sfreq: required only if raw is numpy array.

	spatial: None | 'car' | 'laplacian'
		Spatial filter type.

	spatial_ch: None | list (for CAR) | dict (for LAPLACIAN)
		Reference channels for spatial filtering. May contain channel names.
		'car': channel indices used for CAR filtering. If None, use all channels except
			   the trigger channel (index 0).
		'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
		*** Note ***
		Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.

	spectral: None | [l_freq, h_freq]
		Spectral filter.
		if l_freq is None: lowpass filter is applied.
		if h_freq is None: highpass filter is applied.
		otherwise, bandpass filter is applied.

	spectral_ch: None | list
		Channel picks for spectra filtering. May contain channel names.

	notch: None | float | list of frequency in floats
		Notch filter.

	notch_ch: None | list
		Channel picks for notch filtering. May contain channel names.

	multiplier: float
		If not 1, multiply data values excluding trigger values.

	ch_names: None | list
		If raw is numpy array and channel picks are list of strings, ch_names will
		be used as a look-up table to convert channel picks to channel numbers.


	Output
	------
	True if no error.

	"""

	# Check datatype
	if type(raw) == np.ndarray:
		# Numpy array: assume we don't have event channel
		data= raw
		assert sfreq is not None and sfreq > 0, 'Wrong sfreq value.'
		n_channels= data.shape[0]
		eeg_channels= list(range(n_channels))
	else:
		# MNE Raw object: exclude event channel
		ch_names= raw.ch_names
		data= raw._data
		sfreq= raw.info['sfreq']
		n_channels= data.shape[0]
		eeg_channels= list(range(n_channels))
		tch= find_event_channel(raw)
		if tch is None:
			qc.print_c('preprocess(): No trigger channel found. Ignoring.', 'W')
		else:
			tch_name= ch_names[tch]
			eeg_channels.pop(tch)

	# Do unit conversion
	if multiplier != 1:
		data[eeg_channels] *= multiplier

	# Apply spatial filter
	if spatial is None:
		pass
	elif spatial=='car':
		if spatial_ch is None:
			spatial_ch_i= eeg_channels
		elif type( spatial_ch[0] )==str:
			assert ch_names is not None, 'preprocess(): ch_names must not be None'
			spatial_ch_i= [ ch_names.index(c) for c in spatial_ch ]
		else:
			spatial_ch_i= spatial_ch
		data[spatial_ch_i] -= np.mean( data[spatial_ch_i], axis=0 )
	elif spatial=='laplacian':
		if type(spatial_ch) is not dict:
			raise RuntimeError('For Lapcacian, spatial_ch must be of form {CHANNEL:[NEIGHBORS], ...}')
		if type( spatial_ch.keys()[0] )==str:
			spatial_ch_i= {}
			for c in spatial_ch:
				ref_ch= ch_names.index(c)
				spatial_ch_i[ref_ch]= [ ch_names.index(n) for n in spatial_ch[c] ]
		else:
			spatial_ch_i= spatial_ch
		rawcopy= data.copy()
		for src in spatial_ch:
			nei= spatial_ch[src]
			data[src]= rawcopy[src] - np.mean( rawcopy[nei], axis=0 )
	else:
		raise RuntimeError('Unknown spatial filter %s'% spatial)

	# Apply spectral filter
	if spectral is not None:
		if spectral_ch is None:
			spectral_ch= eeg_channels
		elif type( spectral_ch[0] )==str:
			assert ch_names is not None, 'preprocess(): ch_names must not be None'
			spectral_ch_i= [ ch_names.index(c) for c in spectral_ch ]
		else:
			spectral_ch_i= spectral_ch

		if spectral[0] is None:
			mne.filter.low_pass_filter(data, Fs=sfreq, Fp=spectral[1],
				picks=spectral_ch, method='fft', copy=False, verbose='ERROR')
		elif spectral[1] is None:
			mne.filter.high_pass_filter(data, Fs=sfreq, Fp=spectral[0],
				picks=spectral_ch, method='fft', copy=False, verbose='ERROR')
		else:
			mne.filter.band_pass_filter(data, Fs=sfreq, Fp1=spectral[0], Fp2=spectral[1],
				picks=spectral_ch, method='fft', copy=False, verbose='ERROR')

	# Apply notch filter
	if notch is not None:
		if notch_ch is None:
			notch_ch= eeg_channels
		elif type( notch_ch[0] )==str:
			assert ch_names is not None, 'preprocess(): ch_names must not be None'
			notch_ch_i= [ ch_names.index(c) for c in notch_ch ]
		else:
			notch_ch_i= notch_ch

		mne.filter.notch_filter(data, Fs=sfreq, freqs=notch, notch_widths=5,
			picks=notch_ch, method='fft', n_jobs=mp.cpu_count(), copy=False)

	return True


def load_raw(rawfile, spfilter=None, spchannels=None, events_ext=None, multiplier=1, verbose='ERROR'):
	"""
	Loads data from a fif-format file.
	You can convert non-fif files (.eeg, .bdf, .gdf, .pcl) to fif format.

	Parameters:
	rawfile: (absolute) data file path
	spfilter: 'car' | 'laplacian' | None
	spchannels: None | list (for CAR) | dict (for LAPLACIAN)
		'car': channel indices used for CAR filtering. If None, use all channels except
			   the trigger channel (index 0).
		'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
		*** Note ***
		Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.
	events_ext: Add externally recorded events.
				[ [sample_index1, 0, event_value1],... ]
	multiplier: Multiply all values except triggers (to convert unit).

	Returns:
	raw: mne.io.RawArray object. First channel (index 0) is always trigger channel.
	events: mne-compatible events numpy array object (N x [frame, 0, type])
	spfilter= {None | 'car' | 'laplacian'}

	"""

	if not os.path.exists(rawfile):
		qc.print_c('# ERROR: File %s not found'% rawfile, 'r')
		sys.exit(-1)

	extension= rawfile.split('.')[-1]
	assert extension in ['fif','fiff'], 'only fif format is supported'
	raw= mne.io.Raw(rawfile, preload=True, proj=False, verbose=verbose, add_eeg_ref=False)
	preprocess(raw, spatial=spfilter, spatial_ch=spchannels, multiplier=multiplier)

	tch= find_event_channel(raw)
	if tch is not None:
		events= mne.find_events(raw, stim_channel=raw.ch_names[tch], shortest_event=1, uint_cast=True, consecutive=True)
	else:
		events= []

	return raw, events


def load_multi(flist, spfilter=None, spchannels=None, multiplier=1):
	"""
	Load multiple data files and concatenate them into a single series

	- Assumes same sampling rate.
	- Event locations are updated accordingly with new offset.
	- SUpports different number of channels across recordings. In this case, only
	channels common to all recordings will be kept.

	See load_raw() for more details.

	"""

	if len(flist) == 0:
		raise RuntimeError('The file list is empty.')
	elif len(flist)==1:
		return load_raw(flist[0], spfilter=spfilter, spchannels=spchannels, multiplier=multiplier)

	rawlist= []
	events= []
	signals= None
	chset= []
	for f in flist:
		raw, _= load_raw(f, spfilter=spfilter, spchannels=spchannels, multiplier=multiplier)
		rawlist.append(raw)
		chset.append(set(raw.ch_names))

	# find common channels
	ch_common= chset[0]
	for c in range(1, len(chset)):
		ch_common -= chset[c] ^ ch_common

	# move trigger channel to index 0
	ch_common= list(ch_common)
	for i, c in enumerate(ch_common):
		if 'TRIGGER' in c or 'STI ' in c:
			del ch_common[i]
			ch_common.insert(0, 'TRIGGER')
			trigch= 0
			break
	else:
		trigch= None
	
	# concatenate signals
	for raw in rawlist:
		picks= [ raw.ch_names.index(c) for c in ch_common ]
		if signals is None:
			signals= raw._data[picks]
		else:
			signals= np.concatenate( (signals, raw._data[picks]), axis=1 ) # append samples
	
	# create a concatenated raw object and update channel names
	raw= rawlist[0]
	if trigch is None:
		ch_types= ['eeg'] * len( ch_common )
	else:
		ch_types= ['stim'] + ['eeg'] * (len( ch_common ) - 1)
	info= mne.create_info(ch_common, raw.info['sfreq'], ch_types, montage='standard_1005')
	raws= mne.io.RawArray( signals, info )

	# re-calculate event positions
	events= mne.find_events( raws, stim_channel='TRIGGER', shortest_event=1, consecutive=True )

	return raws, events

def butter_bandpass(highcut, lowcut, fs, num_ch):
	"""
	Calculation of bandpass coefficients.
	Order is computed automatically.
	Note that if filter is unstable this function crashes.
	
	TODO: handle problems
	"""

	low = lowcut/(0.5*fs)
	high = highcut/(0.5*fs)
	ord = buttord(high, low, 2, 40)
	b, a = butter(2, [low, high], btype='band')
	zi = np.zeros([a.shape[0]-1, num_ch])
	return b, a, zi

def search_lsl(ignore_markers=False):
	import pylsl, time

	# look for LSL servers
	amp_list= []
	amp_list_backup= []
	while True:
		streamInfos= pylsl.resolve_streams()
		if len(streamInfos) > 0:
			for index, si in enumerate(streamInfos):
				amp_serial= pylsl.StreamInlet(si).info().desc().child('acquisition').child_value('serial_number').strip()
				amp_name= si.name()
				if 'Markers' in amp_name:
					amp_list_backup.append( (index, amp_name, amp_serial) )
				else:
					amp_list.append( (index, amp_name, amp_serial) )
			break
		print('No server available yet on the network...')
		time.sleep(1)

	if ignore_markers is False:
		amp_list += amp_list_backup

	qc.print_c('-- List of servers --', 'W')
	for i, (index, amp_name, amp_serial) in enumerate(amp_list):
		if amp_serial=='': amp_ser= 'N/A'
		else: amp_ser= amp_serial
		qc.print_c( '%d: %s (Serial %s)'% (i, amp_name, amp_ser), 'W' )

	if len(amp_list)==1:
		index= 0
	else:
		index= raw_input('Amp index? Hit enter without index to select the first server.\n>> ')
		index= int( index.strip() )
	amp_index, amp_name, amp_serial= amp_list[index]
	si= streamInfos[amp_index]
	assert amp_name == si.name()
	assert amp_serial == pylsl.StreamInlet(si).info().desc().child('acquisition').child_value('serial_number').strip()
	print('Selected %s (Serial: %s)'% (amp_name, amp_serial))

	return amp_name, amp_serial

def lsl_channel_list(inlet):
	"""
	Reads XML description of LSL header and returns channel list

	Input:
		pylsl.StreamInlet object
	Returns:
		ch_list: [ name1, name2, ... ]
	"""
	ch_list= []

	import xmltodict
	xml=inlet.info().as_xml()
	doc=xmltodict.parse(xml)
	channels= doc['info']['desc']['channels']['channel']
	for ch in channels:
		ch_list.append( ch['label'] )

	return ch_list


def channel_names_to_index(raw, channel_names=None):
	"""
	Return channel indicies among EEG channels
	"""
	if channel_names == None:
		picks= mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	else:
		picks= []
		for c in channel_names:
			if type(c)==int:
				picks.append(c)
			elif type(c)==str:
				if c not in raw.ch_names:
					raise IndexError('Channel %s not found in raw.ch_names'%c)
				picks.append( raw.ch_names.index(c) )
			else:
				raise RuntimeError('channel_names is unknown format.\nchannel_names=%s'% channel_names)

	return picks
