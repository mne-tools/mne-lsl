from __future__ import print_function, division, unicode_literals

"""
Convert known file format to FIF.
"""

import os
import sys
import mne
import pickle
import numpy as np

from builtins import input
from pathlib import Path

from neurodecode import logger

import neurodecode.utils.io as io
import neurodecode.utils.preprocess as preprocess


mne.set_log_level('ERROR')

#----------------------------------------------------------------------
def any2fif(filename, outdir=None, channel_file=None):
    """
    Generic file format converter to mne.io.raw.
    
    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        Saving directory. If None, it will be the directory of the .pkl file.
    channel_file : str
        The file containing the channels name in case of .gdf/.bdf files.
    """
    p = io.parse_path(filename)

    if p.ext in ['pcl', 'pkl', 'pickle']:
        
        eve_file = '%s/%s.txt' % (p.dir, p.name.replace('raw', 'eve'))
        
        # Remove the stream name from the event file name
        eve_file = eve_file.split('-')
        eve_file.pop(-2)
        eve_file = '-'.join(eve_file)
        
        if os.path.exists(eve_file):
            logger.info('Adding events from %s' % eve_file)
        else:
            logger.info('No SOFTWARE event file %s' % eve_file)
            eve_file = None
        pcl2fif(filename, outdir=outdir, external_event=eve_file)
    
    elif p.ext == 'eeg':
        eeg2fif(filename, outdir=outdir)
    
    elif p.ext == 'edf':
        edf2fif(filename, outdir=outdir)    
    
    elif p.ext == 'bdf':
        bdf2fif(filename, outdir=outdir, channel_file=channel_file)
    
    elif p.ext == 'gdf':
        gdf2fif(filename, outdir=outdir, channel_file=channel_file)
    
    elif p.ext == 'xdf':
        xdf2fif(filename, outdir=outdir)
    
    else:  # unknown format
        logger.error('Ignored unrecognized file extension %s. It should be [.pickle | .eeg | .edf | .gdf | .bdf | .xdf]' % p.ext)

#----------------------------------------------------------------------
def pcl2fif(filename, outdir=None, external_event=None, precision='single', replace=False):
    """
    Convert NeuroDecode Python pickle format to mne.io.raw.

    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        Saving directory. If None, it will be the directory of the .pkl file.
    external_event : str
        Event file path in text formatm following mne event struct. Each row should be: index 0 event
    precision : str
        Data matrix format. [single|double|int|short], 'single' improves backward compatability.
    replace : bool
        If true, previous events will be overwritten by the new ones from the external events file.
    """
    p = io.parse_path(filename)
    
    outdir = _create_saving_dir(outdir, p.dir)
    
    fiffile = outdir + p.name + '.fif'
        
    # Load from file
    data = io.load_obj(filename)
    
    # mne format
    raw = _format_pkl_to_mne_RawArray(data)
    
    # Add events from txt file
    if external_event is not None:
        events_index = event_timestamps_to_indices(raw.times, external_event, data["timestamps"][0])
        _add_events_from_txt(raw, events_index, stim_channel='TRIGGER', replace=replace)
    
    # Save
    raw.save(fiffile, verbose=False, overwrite=True, fmt=precision)
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info["ch_names"])

#----------------------------------------------------------------------
def edf2fif(filename, outdir=None):
    """
    Convert European Data Format (EDF) to mne.io.raw.
    
    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        If None, it will be the subdirectory of the fif file.
    """
    p = io.parse_path(filename)
    
    if outdir is None:
        outdir = p.dir
    elif outdir[-1] != '/':
        outdir += '/'

    fiffile = outdir + p.name + '-raw.fif'
    eventsfile = outdir + p.name + '-events_id.pkl'

    # Load the data
    raw = mne.io.read_raw_edf(filename, preload=True)
    
    # Add TRIGGER channel
    _add_empty_trig_ch(raw)

    # Add events to TRIGGER channel
    _events_from_annotations(raw, eventsfile)

    # save and close
    raw.save(fiffile, verbose=False, overwrite=True, fmt='double')
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info['ch_names'])  

#----------------------------------------------------------------------
def bdf2fif(filename, outdir=None, channel_file=None):
    """
    Convert BioSemi (bdf) format to mne.io.raw.
    
    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        If None, it will be the subdirectory of the fif file.
    channel_file : str
        The .txt file containing one channel's name per line
    """
    p = io.parse_path(filename)
    
    if outdir is None:
        outdir = p.dir
    elif outdir[-1] != '/':
        outdir += '/'

    fiffile = outdir + p.name + '.fif'
    
    # Load the data
    raw = mne.io.read_raw_edf(filename, preload=True)
    
    # Add channels names if file is provided
    if channel_file:
        _ch_names_from_file(raw, channel_file)    
    
    # Find trig channel
    trig_ch_guess = _find_trig_ch(raw)
    _move_trig_ch_to_zero(raw, trig_ch_guess)    
    
    # save and close
    raw.save(fiffile, verbose=False, overwrite=True, fmt='double')
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info['ch_names'])      

#----------------------------------------------------------------------
def gdf2fif(filename, outdir=None, channel_file=None):
    """
    Convert g.Tec format (gdf) to mne.io.raw.
    
    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        If None, it will be the subdirectory of the fif file.
    channel_file : str
        The .txt file containing one channel's name per line
    """
    p = io.parse_path(filename)
    if outdir is None:
        outdir = p.dir
    elif outdir[-1] != '/':
        outdir += '/'

    fiffile = outdir + p.name + '-raw.fif'
    eventsfile = outdir + p.name + '-events_id.pkl'

    # Load the data
    raw = mne.io.read_raw_gdf(filename, preload=True)
    
    # Add channels names if file is provided
    if channel_file:
        _ch_names_from_file(raw, channel_file)
    
    # Trigger channel
    trig_ch_guess = _find_trig_ch(raw)
    _move_trig_ch_to_zero(raw, trig_ch_guess)
    
    # Add events from annotations to TRIGGER channel
    _events_from_annotations(raw, eventsfile)    
    
    # save and close
    raw.save(fiffile, verbose=False, overwrite=True, fmt='double')
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info['ch_names'])

#----------------------------------------------------------------------
def xdf2fif(filename, outdir=None):
    """
    Convert LabStreamingLayer format (xdf) to mne.io.raw.
    
    Note: Can only convert data with one eeg stream.
    
    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format. 
    outdir : str
        If None, it will be the subdirectory of the fif file.
    """
    from pyxdf import pyxdf

    p = io.parse_path(filename)
    
    if outdir is None:
        outdir = p.dir
    elif outdir[-1] != '/':
        outdir += '/'

    fiffile = outdir + p.name + '-raw.fif'

    # Load the data
    data = pyxdf.load_xdf(filename)         # channel x times
    raw_data = data[0][0]['time_series'].T  # times x channel
    
    # 99% of plugins stream uV, but mne expect V
    raw_data *= 10 ** - 6
    
    #  Extract stream info
    labels, types, units = _get_ch_info_xdf(data[0][0])
    sample_rate = int(data[0][0]['info']['nominal_srate'][0])
    
    # Find trig channel
    trig_ch_guess = preprocess.find_event_channel(None, labels)

    if trig_ch_guess is None:
        logger.warning('No trigger channel found: {}'.format(labels))
        trig_ch_guess = int(input('Provide the trigger channel index to continue: \n >> '))
    
    # Plugins use other name (markers....)
    types[trig_ch_guess ]= 'stim'
    
    # Create MNE raw struct
    info = mne.create_info(labels, sample_rate, types)
    raw = mne.io.RawArray(raw_data, info)
    
    _move_trig_ch_to_zero(raw, trig_ch_guess)

    # save and close
    raw.save(fiffile, verbose=False, overwrite=True, fmt='double')
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info['ch_names'])    

#----------------------------------------------------------------------
def eeg2fif(filename, outdir=None):
    """
    Convert Brain Products EEG format to mne.io.raw.
    
    Parameters
    ----------
    filename : str
        The .eeg file path
    outdir : str
        The folder where the fif file will be saved.
    """
    p = io.parse_path(filename)
    
    if outdir is None:
        outdir = p.dir
    elif outdir[-1] != '/':
        outdir += '/'

    headerfile = p.dir + p.name + '.vhdr'
    fiffile = outdir + p.name + '-raw.fif'
    eventsfile = outdir + p.name + '-events_id.pkl'
    
    # Load from file
    raw = mne.io.read_raw_brainvision(headerfile, preload=True)    

    # Add TRIGGER channel
    _add_empty_trig_ch(raw)
    _move_trig_ch_to_zero(raw, -1)

    # Add events to TRIGGER channel
    _events_from_annotations(raw, eventsfile)

    # save and close
    raw.save(fiffile, verbose=False, overwrite=True, fmt='double')
    logger.info('Data saved to: {}'.format(fiffile))

    _saveChannels2txt(outdir, raw.info['ch_names'])

#----------------------------------------------------------------------
def event_timestamps_to_indices(raw_timestamps, eventfile, offset):
    """
    Convert LSL timestamps to sample indices for separetely recorded events.

    Parameters
    ----------
    raw_timestamps : list
        The whole data's timestamps (mne: start at 0.0 sec)
    eventfile : str
        Event file containing the events, indexed with LSL timestamps
    offset : float
        The first sample's LSL timestamp, to start at 0.0 sec 

    Returns
    -------
    np.array
        The events [shape=(n_events, 3)]; used as input to mne.io.RawArray.add_events().
    """

    ts_min = min(raw_timestamps)
    ts_max = max(raw_timestamps)
    events = []

    with open(eventfile) as f:
        for l in f:
            data = l.strip().split('\t')
            event_ts = float(data[0]) - offset
            event_value = int(data[2])
            next_index = np.searchsorted(raw_timestamps, event_ts) #  first index not smaller than raw_timestamps
            if next_index >= len(raw_timestamps):
                logger.warning('Event %d at time %.3f is out of time range (%.3f - %.3f).' % (event_value, event_ts, ts_min, ts_max))
            else:
                events.append([next_index, 0, event_value])
    
    return events

#----------------------------------------------------------------------
def _add_events_from_txt(raw, events_index, stim_channel='TRIGGER', replace=False):
    """
    Merge the events extracted from a txt file to the trigger channel.
    
    Parameters
    ----------
    raw : mne.io.raw
        The mne raw data structure
    events_index : list
        The mne struct events [shape=(n_events, 3)]
    stim_channel : str
        The stin channel to add the events
    replace : bool
        If True the old events on the stim channel are removed before adding the new ones
    """    
    if len(events_index) == 0:
        logger.warning('No events were found in the event file')
    else:
        logger.info('Found %d events' % len(events_index))
        raw.add_events(events_index, stim_channel=stim_channel, replace=replace)    
    
#----------------------------------------------------------------------
def _format_pkl_to_mne_RawArray(data):
    """
    Format the raw data to the mne rawArray structure.
    
    Data must be recorded with NeuroDecode StreamRecorder.
    
    Parameters
    ----------
    data : dict
        Data loaded from the pcl file
    
    Returns
    -------
    mne.io.raw
        The mne raw structure
    """

    if type(data['signals']) == list:
        signals_raw = np.array(data['signals'][0]).T    # to channels x samples
    else:
        signals_raw = data['signals'].T                 # to channels x samples
    
    sample_rate = data['sample_rate']
    
    # Look for channels name
    if 'ch_names' not in data:
        ch_names = ['CH%d' % (x + 1) for x in range(signals_raw.shape[0])]
    else:
        ch_names = data['ch_names']

    # search for the trigger channel
    trig_ch = preprocess.find_event_channel(signals_raw, ch_names)

    # move trigger channel to index 0
    if trig_ch is None:
        # assuming no event channel exists, add a event channel to index 0 for consistency.
        logger.warning('No event channel was not found. Adding a blank event channel to index 0.')
        eventch = np.zeros([1, signals_raw.shape[1]])
        signals = np.concatenate((eventch, signals_raw), axis=0)
        num_eeg_channels = signals_raw.shape[0] # data['channels'] is not reliable any more
        trig_ch = 0
        ch_names = ['TRIGGER'] + ch_names
    
    elif trig_ch == 0:
        signals = signals_raw
        num_eeg_channels = data['channels'] - 1
    
    else:
        # move event channel to 0
        logger.info('Moving event channel %d to 0.' % trig_ch)
        signals = np.concatenate((signals_raw[[trig_ch]], signals_raw[:trig_ch], signals_raw[trig_ch + 1:]), axis=0)
        assert signals_raw.shape == signals.shape
        num_eeg_channels = data['channels'] - 1
        ch_names.pop(trig_ch)
        trig_ch = 0
        ch_names.insert(trig_ch, 'TRIGGER')
        logger.info('New channel list:')
        for c in ch_names:
            logger.info('%s' % c)

    ch_info = ['stim'] + ['eeg'] * num_eeg_channels
    info = mne.create_info(ch_names, sample_rate, ch_info)

    # create Raw object
    raw = mne.io.RawArray(signals, info)
    
    return raw

#----------------------------------------------------------------------
def _events_from_annotations(raw, eventsfile):
    """
    Extract events from annotations, add them to TRIGGER channel and
    save the created dict to pkl file.
    
    If old events on the TRIGGER channel, they will be removed.
    
    Parameters
    ----------
    raw : mne.io.raw
        The mne raw data.
    eventsfile : str
        The .pkl file path for saving the str-int dictionnary.
    """
    # Extract events from annotations
    events, events_id = mne.events_from_annotations(raw)
    
    # add events
    if events.any():
        raw.add_events(events, stim_channel='TRIGGER', replace=True)
        
        # Save dict to .pkl file
        with open(eventsfile, 'wb') as handle:
            pickle.dump(events_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Annotations-events dict saved to %s' % eventsfile)
    else:
        logger.warning('No annotations/events found.')
        
#----------------------------------------------------------------------
def _add_empty_trig_ch(raw):
    """
    Add to mne raw an empty TRIGGER channel at position 0.
    
    Parameters
    ----------
    raw : mne.io.raw
        The mne raw data.
    """
    # Add empty event channel
    eventdata = np.zeros([1, raw.get_data().shape[1]])
    info = mne.create_info(['TRIGGER'], raw.info['sfreq'], ['stim'])
    eventch = mne.io.RawArray(eventdata, info)

    # Ensure same info as raw.info
    eventch.info['lowpass'] = raw.info['lowpass']
    eventch.info['highpass'] = raw.info['highpass']
    raw.add_channels([eventch])

#----------------------------------------------------------------------
def _move_trig_ch_to_zero(raw, trig_pos):
    """
    Move trig channel at position 0
    """
    raw.rename_channels({raw.info["ch_names"][trig_pos]: 'TRIGGER',})
    raw.set_channel_types({raw.info["ch_names"][trig_pos]: 'stim',})
    
    ch_names = list(raw.info["ch_names"])
    ch_names.pop(trig_pos)
    ch_names.insert(0, 'TRIGGER')
    
    raw.reorder_channels(ch_names)    


#----------------------------------------------------------------------
def _find_trig_ch(raw):
    """
    Find the trigger channel from the raw.info['ch_names']
    
    Parameters
    ----------
    raw : mne.io.raw
        The mne raw struture.
        
    Returns:
    --------
    int : the trigger channel index.
    """
    trig_ch_guess = preprocess.find_event_channel(raw)
    
    if trig_ch_guess is None:
        logger.warning('No trigger channel found: {}'.format(raw.info['ch_names']))
        trig_ch_guess = int(input('Provide the trigger channel index to continue: \n >> '))
    
    return trig_ch_guess

#----------------------------------------------------------------------
def _ch_names_from_file(raw, channel_file):
    """
    Change the channels names according to the file
    
    Parameters
    ----------
    raw : mne.io.raw
        The mne raw struture.
    channel_file : str
        The path to the .txt file containing the channels name.
    """
    ch_names_file = []
    
    for l in open(channel_file):
        ch_names_file.append(l.strip())
    
    if len(ch_names_file) != len(raw.info['ch_names']):
        raise RuntimeError("The channel file does not contain the same number of channels than the original data.")
    
    ch_names = {}
    for i in range(len(ch_names_file)):
        ch_names[raw.info['ch_names'][i]] = ch_names_file[i]
    
    raw.rename_channels(ch_names)
    
# ------------------------------------------------------------------------------------------
def _get_ch_info_xdf(stream):
    """
    Extract the info for each eeg channels (label, type and unit)
    """
    labels, types, units = [], [], []
    n_chans = int(stream["info"]["channel_count"][0])
    
    # Get channels labels, types and units
    if stream["info"]["desc"] and stream["info"]["desc"][0]:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            try:
                types.append(ch["type"][0].lower())
                units.append(ch["unit"][0])
            except:
                pass
    
    if not labels:
        labels = ["ch_" + str(n) for n in range(n_chans)]
    if not types:
        units = ["eeg" for _ in range(n_chans)]    
    if not units:
        units = ["uV" for _ in range(n_chans)]    
    
    return labels, types, units

#----------------------------------------------------------------------
def _create_saving_dir(outdir, fdir):
    """
    Create the directory where the .fif file will be saved
    
    Parameters
    ----------
    outdir : str
        Saving directory. If None, it will be the directory of the .pkl file.
    
    """
    if outdir is None:
        outdir = fdir + '/fif/'
    if outdir[-1] != '/':
        outdir += '/'
    io.make_dirs(outdir)
        
    return outdir

#----------------------------------------------------------------------
def _saveChannels2txt(outdir, ch_names):
    """
    Save the channels list to a txt file for the GUI
    """

    filename = outdir + "channelsList.txt"
    config = Path(filename)

    if config.is_file() is False:
        file = open(filename, "w")
        for x in range(len(ch_names)):
            file.write(ch_names[x] + "\n")
        file.close()

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    channel_file = None
    
    if len(sys.argv) > 3:
        raise IOError("Two many arguments provided, max is 2  (directory and channels file)")
    
    if len(sys.argv) > 2:
        channel_file = sys.argv[2]
    
    if len(sys.argv) > 1:
        if not channel_file:
            channel_answer = input('Channels file (only for .gdf .bdf)? [y/n]\n>> ')
            if channel_answer in ['y', 'Y', 'yes', 'YES', 'Yes']:
                channel_file = input('Provide its path?\n>> ')
                
        input_dir = sys.argv[1]
    
    if len(sys.argv) == 1:
        input_dir = input('Input directory?\n>> ')
        
        channel_answer = input('Channels file (only for .gdf .bdf)? [y/n]\n>> ')
        if channel_answer in ['y', 'Y', 'yes', 'YES', 'Yes']:
            channel_file = input('Provide its path?\n>> ')
    
    count = 0
    for f in io.get_file_list(input_dir, fullpath=True, recursive=True):
        p = io.parse_path(f)
        outdir = p.dir + '/fif/'
        
        if p.ext in ['pcl', 'pkl', 'pickle', 'bdf', 'edf', 'gdf', 'eeg', 'xdf']:
            logger.info('Converting %s' % f)
            any2fif(f, outdir=outdir, channel_file=channel_file)
            count += 1
    
            logger.info('%d files converted.' % count)
