from __future__ import print_function, division

"""
Convert BrainProducts EEG format to standard FIFF format.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

# set your file name (without extension) here
BASENAME= r'D:\data\fNIRS\q5\20160307\fnirs_eeg1'

def eeg2fif(basename):
	import sys, os
	import scipy.io
	import numpy as np
	import pycnbi_config, mne
	import q_common as qc

	eegfile= BASENAME + '.eeg'
	matfile= BASENAME + '.mat'
	markerfile= BASENAME + '.vmrk'
	fiffile= BASENAME + '.fif'

	# convert to mat using MATLAB
	if not os.path.exists(matfile):
		print('Converting input to mat file')
		run= "[sig,header]=sload('%s'); save('%s','sig','header');"%(eegfile,matfile)
		qc.matlab(run)
		if not os.path.exists(matfile):
			qc.print_c('>> ERROR: mat file convertion error.', 'r')
			sys.exit()
	else:
		print('MAT file already exists. Skipping conversion.')

	# extract events
	events= []
	for l in open(markerfile):
		if 'Stimulus,S' in l:
			#event, sample_index= l.split('  ')[-1].split(',')[:2]
			data= l.split(',')[1:3]
			event= int( data[0][1:] ) # ignore 'S'
			sample_index= int( data[1] )
			events.append( [sample_index, 0, event] )

	# load data and create fif header
	mat= scipy.io.loadmat(matfile)
	headers= mat['header']
	sample_rate= int( mat['header']['SampleRate'] )
	signals= mat['sig'].T # channels x samples
	nch, t_len= signals.shape
	ch_names= ['TRIGGER'] + ['CH%d'% (x+1) for x in range(nch)]
	ch_info= ['stim'] + ['eeg'] * (nch)
	info= mne.create_info( ch_names, sample_rate, ch_info )

	# add event channel
	eventch= np.zeros( [1, signals.shape[1]] )
	signals= np.concatenate( (eventch, signals), axis=0 )

	# create Raw object
	raw= mne.io.RawArray( signals, info )

	# add events
	raw.add_events(events, 'TRIGGER')

	# save and close
	raw.save(fiffile, verbose=False, overwrite=True)
	print('Done.')

if __name__=='__main__':
	eeg2fif(BASENAME)
