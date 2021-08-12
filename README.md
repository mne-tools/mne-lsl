
# Introduction

**NeuroDecode** provides a real-time brain signal decoding framework. The original version developped by [**Kyuhwa Lee**](https://github.com/dbdq) was recognised at [Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding) with the First Prize Award (2016) after achieving high decoding accuracy.

The underlying data communication is based on Lab Streaming Layer (LSL) which provides sub-millisecond time synchronization accuracy. Any signal acquisition system supported by native LSL or OpenVibe is also supported by Neurodecode. Since the data communication is based on TCP, signals can be transmitted wirelessly. For more information about LSL, please visit the [LSL github](https://github.com/sccn/labstreaminglayer).

This fork is based on the refactor version by [**Arnaud Desvachez**](https://github.com/dnastars) for the [Fondation Campus Biotech Geneva (FCBG)](https://github.com/fcbg-hnp). The low-level functionnalities have been reworked and improved, while the decoding functionnalities have been dropped.

# Installation
NeuroDecode supports `python >= 3.6` and requires:
- numpy
- scipy
- pylsl
- mne
- pyqt5
- pyqtgraph

Optional dependencies for trigger via an [Arduino to LPT converter](https://github.com/fcbg-hnp/arduino-trigger):
- pyserial

Optional dependencies for StreamViewer alternative backends:
- vispy

NeuroDecode can be installed in normal mode with `python setup.py install` or in developement mode with `python setup.py develop`. Optional dependencies can be installed using the keywords:
- trigger_arduino2lpt
- vispy_backend

TODO:
- [ ] Add NeuroDecode to Pypi for pip install
- [ ] Add NeuroDecode to conda-forge for conda install

# Documentation
NeuroDecode is centered around 4 main modules: `stream_receiver`, `stream_recorder`, `stream_player` and `stream_viewer`.

## StreamReceiver
The stream receiver connects to one or more LSL streams and acquires data from those. Supported streams are:
- EEG
- Markers

**Example:**
```
from neurodecode.stream_receiver import StreamReceiver

# Connects to all available streams
sr = StreamReceiver(bufsize=1, winsize=1, stream_name=None)
# Update each stream buffer with new data
sr.acquire()
# Retrieve buffer/window for the stream named 'StreamPlayer'
data, timestamps = sr.get_window(stream_name='StreamPlayer')
```
The data and its timestamps are returned as numpy array:
- `data.shape = (samples, channels)`
- `timestamps.shape = (samples, )`

## StreamRecorder
The stream recorder connects to one or more LSL streams and periodically acquires data from those until stopped, and then saves the acquired data to disk in pickle `.pcl` and in FIF `.fif` format.

**Example:**
```
import time
from neurodecode.stream_recorder import StreamRecorder

# Connects to all available streams
recorder = StreamRecorder(record_dir=None, fname=None, stream_name=None)
recorder.start(verbose=True)
time.sleep(10)
recorder.stop()
```
When the argument `record_dir` is set to None, the current folder obtained with `pathlib.Path.cwd()` is used.
When the argument `fname` is set to None, the created files' stem use the start datetime.

**CLI:** The stream recorder can be called by command-line in a terminal by using either `nd stream_recorder` or `nd_stream_recorder` followed by the optional arguments `-d`, `-f`, `-s` respectively for `record_dir`, `fname`, and `stream_name`.
```
nd_stream_recorder -d "D:/Data"
nd_stream_recorder -d "D:/Data" -f test
nd_stream_recorder -d "D:/Data" -f test -s openvibeSignals
```
## StreamPlayer
The stream player loads a previously recorded `.fif` file and creates a LSL server streaming data from this file. The stream player can be used to test code with a fake LSL data stream.

**Example:**
```
import time
from neurodecode.stream_player import StreamPlayer

sp = StreamPlayer(stream_name='StreamPlayer', fif_file=r'path to .fif')
sp.start()
time.sleep(10)
sp.stop()
```
**CLI:**  The stream player can be called by command-line in a terminal by using either `nd stream_player` or `nd_stream_player` followed by positional arguments `stream_name` and `fif_file` and the optional arguments `-c` and `-t` respectively for `chunk_size` and `trigger_file`.
```
nd_stream_player StreamPlayer "D:/Data/data-raw.fif"
nd_stream_player StreamPlayer "D:/Data/data-raw.fif" -c 16
nd_stream_player StreamPlayer "D:/Data/data-raw.fif" -c 16 -t "D:/triggerdef.ini"
```
## StreamViewer
The stream viewer creates a 2-window GUI composed of a control GUI and a plotter GUI to display the data acquired from an LSL server in real-time.

**CLI:** The stream viewer can be called by command-line in a terminal by using either `nd stream_viewer` or `nd_stream_viewer` followed by the optional argument `-s` and `-b` respectively for the `stream_name` and `backend`. If no stream name is provided, a prompt will ask the user to select the desired non-marker stream to display. The supported backends are `pyqt5` (default) and `vispy` (incomplete).
```
nd_stream_viewer
nd_stream_viewer -s StreamPlayer
nd_stream_viewer -s StreamPlayer -b vispy
```

## Triggers
Triggers includes functions to mark time event by sending a trigger which will be saved on the *TRIGGER* channel of the on-going recording. Triggers can be achieved either through hardware or through software.

Currently, the supported hardware triggers use an LPT port.

**Example:**
```
import time
from neurodecode.stream_recorder import StreamRecorder
from neurodecode.triggers.software import TriggerSoftware
from neurodecode.triggers.lpt import TriggerArduino2LPT

# Software trigger
recorder = StreamRecorder()
recorder.start()
trigger = TriggerSoftware(recorder)
for k in range(1, 5):
    trigger.signal(k)
    time.sleep(1)
trigger.close()
recorder.stop()

# Hardware trigger through Arduino LPT converter
recorder = StreamRecorder()
recorder.start()
trigger = TriggerArduino2LPT()
for k in range(1, 5):
    trigger.signal(k)
    time.sleep(1)
trigger.close()
recorder.stop()
```
Note that closing the trigger before stopping the recording may not be required for all kind of triggers.

# Copyright and license
The codes are released under [GNU Lesser General Public License](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html).
