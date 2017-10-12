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

OUT_DIR = 'D:/data/Records'  # give absolute path

import pycnbi  # load from global common folder
import time, os, sys, datetime
import pycnbi.utils.q_common as qc
import pycnbi.stream_receiver.stream_receiver as receiver
import multiprocessing as mp
import pycnbi.utils.cnbi_lsl as cnbi_lsl
import sys
import pylsl
import numpy as np
from builtins import input


# record start
def record(state, amp_name, amp_serial, eeg_only=False):
    # set data file name
    filename = time.strftime(OUT_DIR + "/%Y%m%d-%H%M%S-raw.pcl", time.localtime())
    qc.print_c('\n>> Output file: %s' % (filename), 'W')

    # test writability
    try:
        qc.make_dirs(OUT_DIR)
        open(filename, 'w').write('The data will written when the recording is finished.')
    except:
        qc.print_c('\n*** ERROR: There was a problem writing file %s\n' % filename, 'W')
        sys.exit(-1)

    # start a server for sending out data filename for software trigger
    outlet = cnbi_lsl.start_server('StreamRecorderInfo', channel_format='string', source_id=filename, stype='Markers')

    # connect to EEG stream server
    sr = receiver.StreamReceiver(amp_name=amp_name, amp_serial=amp_serial, eeg_only=eeg_only)

    # record start
    qc.print_c('\n>> Recording started (PID %d).' % os.getpid(), 'W')
    qc.print_c('\n>> Press Enter to stop recording', 'G')
    tm = qc.Timer(autoreset=True)
    next_sec = 1
    while state.value == 1:
        sr.acquire()
        if sr.get_buflen() > next_sec:
            duration = str(datetime.timedelta(seconds=int(sr.get_buflen())))
            print('RECORDING %s' % duration)
            next_sec += 1
        tm.sleep_atleast(0.01)

    # record stop
    qc.print_c('>> Stop requested. Copying buffer', 'G')
    buffers, times = sr.get_buffer()
    signals = buffers
    events = None

    # channels = total channels from amp, including trigger channel
    data = {'signals':signals, 'timestamps':times, 'events':events,
            'sample_rate':sr.get_sample_rate(), 'channels':sr.get_num_channels(),
            'ch_names':sr.get_channel_names()}
    qc.print_c('Saving data ...', 'W')
    qc.save_obj(filename, data)
    print('Saved to %s\n' % filename)

    qc.print_c('Converting raw file into a fif format.', 'W')
    import convert2fif as cf
    cf.pcl2fif(filename)


if __name__ == '__main__':
    eeg_only = False

    if len(sys.argv) == 2:
        amp_name = sys.argv[1]
        amp_serial = None
    elif len(sys.argv) == 3:
        amp_name, amp_serial = sys.argv[1:3]
    else:
        amp_name, amp_serial = pu.search_lsl(ignore_markers=True)
    if amp_name == 'None':
        amp_name = None
    qc.print_c('Connecting to a server %s (Serial %s)' % (amp_name, amp_serial), 'W')

    qc.print_c('\n>> Press Enter to start recording.', 'G')
    key = input()
    state = mp.Value('i', 1)
    proc = mp.Process(target=record, args=[state, amp_name, amp_serial, eeg_only])
    proc.start()

    input('')
    state.value = 0
    qc.print_c('(main) Waiting for recorder process to finish.', 'W')
    proc.join(10)
    if proc.is_alive():
        qc.print_c('>> ERROR: Recorder process not finihsing. Are you running from Spyder?', 'R')
        qc.shell()

    sys.stdout.flush()
    print('>> Done.')
