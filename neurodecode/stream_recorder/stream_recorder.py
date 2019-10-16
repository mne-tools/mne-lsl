from __future__ import print_function, division

"""
stream_receiver.py
Acquires signals from LSL server and save into buffer.
Command-line arguments:
  #1: AMP_NAME
  #2: AMP_SERIAL (can be omitted if no serial number available)
  If no argument is supplied, you will be prompted to select one
  from a list of available LSL servers.
Example:
  python stream_recorder.py openvibeSignals
TODO:
- Support HDF output.
- Write simulatenously while receivng data.
- Support multiple amps.
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

import os
import sys
import time
import datetime
import numpy as np
import multiprocessing as mp
import neurodecode.utils.add_lsl_events
import neurodecode.utils.q_common as qc
import neurodecode.utils.pycnbi_utils as pu
from neurodecode.utils.convert2fif import pcl2fif
from neurodecode.utils.cnbi_lsl import start_server
from neurodecode.gui.streams import redirect_stdout_to_queue
from neurodecode.stream_receiver.stream_receiver import StreamReceiver
from neurodecode import logger
from builtins import input


def record(recordState, amp_name, amp_serial, record_dir, eeg_only, recordLogger=logger, queue=None):
    
    redirect_stdout_to_queue(recordLogger, queue, 'INFO')
    
    # set data file name
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    pcl_file = "%s/%s-raw.pcl" % (record_dir, timestamp)
    eve_file = '%s/%s-eve.txt' % (record_dir, timestamp)
    recordLogger.info('>> Output file: %s' % (pcl_file))

    # test writability
    try:
        qc.make_dirs(record_dir)
        open(pcl_file, 'w').write('The data will written when the recording is finished.')
    except:
        raise RuntimeError('Problem writing to %s. Check permission.' % pcl_file)

    # start a server for sending out data pcl_file when software trigger is used
    outlet = start_server('StreamRecorderInfo', channel_format='string',\
        source_id=eve_file, stype='Markers')

    # connect to EEG stream server
    sr = StreamReceiver(buffer_size=0, amp_name=amp_name, amp_serial=amp_serial, eeg_only=eeg_only)

    # start recording
    recordLogger.info('\n>> Recording started (PID %d).' % os.getpid())

    with recordState.get_lock():
        recordState.value = 1
        
    tm = qc.Timer(autoreset=True)
    next_sec = 1
    while recordState.value == 1:
        sr.acquire()
        if sr.get_buflen() > next_sec:
            duration = str(datetime.timedelta(seconds=int(sr.get_buflen())))
            recordLogger.info('RECORDING %s' % duration)
            next_sec += 1
        tm.sleep_atleast(0.001)

    # record stop
    recordLogger.info('>> Stop requested. Copying buffer')
    buffers, times = sr.get_buffer()
    signals = buffers
    events = None

    # channels = total channels from amp, including trigger channel
    data = {'signals':signals, 'timestamps':times, 'events':events,
            'sample_rate':sr.get_sample_rate(), 'channels':sr.get_num_channels(),
            'ch_names':sr.get_channel_names(), 'lsl_time_offset':sr.lsl_time_offset}
    recordLogger.info('Saving raw data ...')
    qc.save_obj(pcl_file, data)
    recordLogger.info('Saved to %s\n' % pcl_file)

    # automatically convert to fif and use event file if it exists (software trigger)
    if os.path.exists(eve_file):
        recordLogger.info('Found matching event file, adding events.')
    else:
        eve_file = None
    recordLogger.info('Converting raw file into fif.')
    pcl2fif(pcl_file, external_event=eve_file)

def run(record_dir, amp_name, amp_serial, recordLogger=logger, eeg_only=False, queue=None):
    recordLogger.info('\nOutput directory: %s' % (record_dir))

    # spawn the recorder as a child process
    recordLogger.info('\n>> Press Enter to start recording.')
    key = input()
    recordState = mp.Value('i', 1)
    proc = mp.Process(target=record, args=[recordState, amp_name, amp_serial, record_dir, eeg_only, recordLogger , queue])
    proc.start()

    # clean up
    time.sleep(1) # required on some Python distribution
    input()
    recordState.value = 0
    recordLogger.info('(main) Waiting for recorder process to finish.')
    proc.join(10)
    if proc.is_alive():
        recordLogger.error('Recorder process not finishing. Are you running from Spyder?')
        recordLogger.error('Dropping into a shell')
        qc.shell()
    sys.stdout.flush()
    recordLogger.info('Recording finished.')

# for batch script
def batch_run(record_dir=None, amp_name=None, amp_serial=None):
    # configure LSL server name and device serial if available
    if not record_dir:
        record_dir = '%s/records' % os.getcwd()    
    if not amp_name:
        amp_name, amp_serial = pu.search_lsl(ignore_markers=True)
    run(record_dir, amp_name=amp_name, amp_serial=amp_serial)

def run_gui(recordState, protocolState, record_dir, recordLogger=logger, amp_name=None, amp_serial=None, eeg_only=False, queue=None):
    
    redirect_stdout_to_queue(recordLogger, queue, 'INFO')
    
    # configure LSL server name and device serial if available
    if not amp_name:
        amp_name, amp_serial = pu.search_lsl(recordState, recordLogger, ignore_markers=True)

    recordLogger.info('\nOutput directory: %s' % (record_dir))

    # spawn the recorder as a child process
    recordLogger.info('\n>> Recording started.')
    proc = mp.Process(target=record, args=[recordState, amp_name, amp_serial, record_dir, eeg_only, recordLogger , queue])
    proc.start()
    
    while not recordState.value:
        pass
    
    # Launching the protocol (shared variable)
    with protocolState.get_lock():
        protocolState.value = 1

    # Continue recording until the shared variable changes to 0.
    while recordState.value:
        time.sleep(1)

    recordLogger.info('(main) Waiting for recorder process to finish.')
    proc.join(10)
    if proc.is_alive():
        recordLogger.error('Recorder process not finishing. Are you running from Spyder?')
        recordLogger.error('Dropping into a shell')
        qc.shell()
    sys.stdout.flush()
    recordLogger.info('Recording finished.')


# default sample recorder
if __name__ == '__main__':
    record_dir = None
    amp_name = None
    amp_serial = None
    if len(sys.argv) > 3:
        amp_serial = sys.argv[3]
    if len(sys.argv) > 2:
        amp_name = sys.argv[2]
    if len(sys.argv) > 1:
        record_dir = sys.argv[1]
    batch_run(record_dir, amp_name, amp_serial)