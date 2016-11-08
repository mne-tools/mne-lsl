from __future__ import print_function, division

"""
Time-frequency analysis using Morlet wavelets or multitapers

Kyuhwa Lee, 2015

"""

import pycnbi_config
import pycnbi_utils as pu
import sys, os, mne, scipy
import multiprocessing as mp
import numpy as np
import q_common as qc
from mne.time_frequency import tfr_morlet, tfr_multitaper
from IPython import embed

# which wavelet?
tfr= tfr_multitaper
#tfr= tfr_morlet


def get_tfr(cfg):
	if hasattr(cfg, 'DATA_DIRS'):
		# concatenate multiple files
		for ddir in cfg.DATA_DIRS:
			ddir= ddir.replace('\\','/')
			if ddir[-1] != '/': ddir += '/'
			flist= []
			for f in qc.get_file_list(ddir, fullpath=True, recursive=True):
				[fdir, fname, fext]= qc.parse_path(f)
				if fext in ['fif', 'bdf', 'gdf']:
					flist.append(f)
			raw, events= pu.load_multi(flist)

			# custom events
			if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
				events= mne.read_events(cfg.EVENT_FILE)

			sp= ddir.split('/')
			file_prefix= '-'.join(sp[-4:-1])
			#outpath= '/'.join(sp[:-4])
			outpath= ddir

	else:
		print('Loading', cfg.DATA_FILE)
		raw, events= pu.load_raw(cfg.DATA_FILE)

		# custom events
		if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
			events= mne.read_events(cfg.EVENT_FILE)

		[outpath, file_prefix, _]= qc.parse_path(cfg.DATA_FILE)

	# set channels of interest
	picks= pu.channel_names_to_index(raw, cfg.CHANNEL_PICKS)
	spchannels= pu.channel_names_to_index(raw, cfg.SP_CHANNELS)

	if max(picks) > len(raw.info['ch_names']):
		msg= 'ERROR: "picks" has a channel index %d while there are only %d channels.'%\
			( max(picks),len(raw.info['ch_names']) )
		raise RuntimeError(msg)

	# Apply filters
	pu.preprocess(raw, spatial=cfg.SP_FILTER, spatial_ch=spchannels, spectral=cfg.TP_FILTER,
		spectral_ch=picks, notch=cfg.NOTCH_FILTER, notch_ch=picks, multiplier=cfg.MULTIPLIER)

	# Read epochs
	try:
		classes= {}
		for t in cfg.TRIGGERS:
			if t in set(events[:,-1]):
				if hasattr(cfg, 'tdef'):
					classes[cfg.tdef.by_value[t]]= t
				else:
					classes[str(t)]= t
		assert len(classes) > 0

		epochs_all= mne.Epochs(raw, events, classes, tmin=cfg.EPOCH[0]-0.5, tmax=cfg.EPOCH[1]+0.5,
			proj=False, picks=picks, baseline=None, preload=True, add_eeg_ref=False)
		if epochs_all.drop_log_stats() > 0:
			print('\n** Bad epochs found. Dropping into a Python shell.')
			print( epochs_all.drop_log )
			print('\nType exit to continue.\n')
			embed()
	except:
		import pdb, traceback
		print('\n*** (tfr_export) ERROR OCCURRED WHILE EPOCHING ***')
		traceback.print_exc()
		pdb.set_trace()

	power= {}
	for evname in classes:
		export_dir= '%s/%s'% (outpath,evname)
		qc.make_dirs( export_dir )
		print('\n>> Processing %s'% evname)
		freqs= cfg.FREQ_RANGE  # define frequencies of interest
		n_cycles= freqs / 2.  # different number of cycle per frequency
		if cfg.POWER_AVERAGED:
			epochs= epochs_all[evname][:]
			power[evname]= tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
				return_itc=False, decim=1, n_jobs=mp.cpu_count() )
			power[evname]= power[evname].crop(tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1])

			if cfg.EXPORT_MATLAB is True:
				# MATLAB export
				mout= '%s/%s-%s-%s-ep%02d.jpg'% (export_dir, file_prefix, cfg.SP_FILTER, evname, ep+1)
				scipy.io.savemat( mout, { 'tfr':power[evname].data, 'chs':power[evname].ch_names } )
			else:
				# Inspect power for each channel
				for ch in np.arange(len(picks)):
					chname= raw.ch_names[picks[ch]]
					title= 'Peri-event %s - Channel %s'% (evname,chname)

					# mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
					fig= power[evname].plot( [ch], baseline=cfg.BS_TIMES, mode='logratio', show=False,
						colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False )
					fout= '%s/%s-%s-%s-%s.jpg'% (export_dir, file_prefix, cfg.SP_FILTER, evname, chname)
					print('Exporting to %s'% fout)
					fig.savefig(fout)
		else:
			for ep in range(len(epochs_all[evname])):
				epochs= epochs_all[evname][ep]
				power[evname]= tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
					return_itc=False, decim=1, n_jobs=mp.cpu_count() )
				power[evname]= power[evname].crop(tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1])

				if cfg.EXPORT_MATLAB is True:
					# MATLAB export
					mout= '%s/%s-%s-%s-ep%02d.jpg'% (export_dir, file_prefix, cfg.SP_FILTER, evname, ep+1)
					scipy.io.savemat( mout, { 'tfr':power[evname].data, 'chs':power[evname].ch_names } )
				else:
					# Inspect power for each channel
					for ch in np.arange(len(picks)):
						chname= raw.ch_names[picks[ch]]
						title= 'Peri-event %s - Channel %s, Trial %d'% (evname,chname,ep+1)

						# mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
						fig= power[evname].plot( [ch], baseline=cfg.BS_TIMES, mode='logratio', show=False,
							colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False )
						fout= '%s/%s-%s-%s-%s-ep%02d.jpg'% (export_dir, file_prefix, cfg.SP_FILTER, evname, chname, ep+1)
						fig.savefig(fout)
						print('Exported %s'% fout)

	if hasattr(cfg, 'POWER_DIFF'):
		export_dir= '%s/diff'% outpath
		qc.make_dirs(export_dir)
		labels= classes.keys()
		df= power[labels[0]] - power[labels[1]]
		#df.data= np.abs( df.data )
		df.data= np.log( np.abs( df.data ) )
		# Inspect power diff for each channel
		for ch in np.arange(len(picks)):
			chname= raw.ch_names[picks[ch]]
			title= 'Peri-event %s-%s - Channel %s'% (labels[0],labels[1],chname)

			# mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
			fig= df.plot( [ch], baseline=(None,0), mode='mean', show=False,
				colorbar=True, title=title, vmin=3.0, vmax=-3.0, dB=False )
			fout= '%s/%s-%s-diff-%s-%s-%s.jpg'% (export_dir, file_prefix, cfg.SP_FILTER, labels[0], labels[1], chname)
			print('Exporting to %s'% fout)
			fig.savefig(fout)
	print('Finished !')


if __name__=='__main__':
	import imp
	if len(sys.argv) < 2:
		cfg_module= raw_input('Config file name? ')
	else:
		cfg_module= sys.argv[1]
	cfg= imp.load_source(cfg_module, cfg_module)
	get_tfr(cfg)
