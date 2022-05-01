[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![PyPI Downloads](https://pepy.tech/badge/bsl)](https://pepy.tech/project/bsl)

[![Brain Streaming Layer](https://raw.githubusercontent.com/bsl-tools/bsl/master/doc/_static/icon-with-name/icon-with-name.svg)](https://bsl-tools.github.io/)

**BrainStreamingLayer** provides a real-time brain signal streaming framework.
**BSL** is a wrapper around the python interface to the Lab Streaming Layer
(LSL). **BSL** goal is to simplify the design of a study using the Lab
Streaming Layer which provides sub-millisecond time synchronization accuracy.

Any signal acquisition system supported by native LSL or OpenVibe is also
supported by BSL. Since the data communication is based on TCP, signals can be
transmitted wirelessly. For more information about LSL, please visit the
[LSL github](https://github.com/sccn/labstreaminglayer).

**BSL** is based on **NeuroDecode**. The original version developped by
[**Kyuhwa Lee**](https://github.com/dbdq) was recognised at
[Microsoft Brain Signal Decoding competition](https://github.com/dbdq/microsoft_decoding)
with the First Prize Award (2016) after achieving high decoding accuracy.
**BSL** is based on the refactor version by
[**Arnaud Desvachez**](https://github.com/dnastars) for the
[Fondation Campus Biotech Geneva (FCBG)](https://github.com/fcbg-hnp).
The low-level functionnalities have been reworked and improved, while the
decoding functionnalities have been dropped.

# Installation
BSL supports `python >= 3.7` and requires:
- numpy
- scipy
- mne
- pyqt5
- pyqtgraph

BSL uses `pylsl` to interface with LSL. A version is provided in
`bsl.externals` and should work 'as is' on most systems. A different version
of `pylsl` can be installed and will be automatically selected by BSL if
available.

BSL uses `psychopy` for trigger via an on-board parallel port. A version
including only the `parallel` module is provided in `bsl.externals` and should
work 'as is' on most systems. A different version of `psychopy` can be
installed and will be automatically selected by BSL if available.

Optional dependencies for trigger via a parallel port (LPT):
- pyserial, if the [Arduino to LPT converter](https://github.com/fcbg-hnp/arduino-trigger)
  is used.

BSL can be installed via `pip` with `pip install bsl`.

BSL can be installed from a cloned repository in normal mode with
`python setup.py install` or in developement mode with
`python setup.py develop`. Optional dependencies can be installed using the
keywords:
- parallel, parallelport, lpt
- doc, doc-build, documentation
- test, testing

Note that the optional dependencies do not include the dependencies present in
`bsl.externals`.

# Documentation

[**Documentation website.**](https://bsl-tools.github.io/)

BSL is centered around 4 main modules: `stream_receiver`, `stream_recorder`,
`stream_player` and `stream_viewer`.

## StreamReceiver
The stream receiver connects to one or more LSL streams and acquires data from
those. Supported streams are:
- EEG
- Markers

**Example:**
```
from bsl import StreamReceiver

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

The data can be returned as an MNE raw instance if `return_raw` is set to
`True`.

## StreamRecorder
The stream recorder connects to one or more LSL streams and periodically
acquires data from those until stopped, and then saves the acquired data to
disk in pickle `.pcl` and in FIF `.fif` format.

**Example:**
```
import time
from bsl import StreamRecorder

# Connects to all available streams
recorder = StreamRecorder(record_dir=None, fname=None, stream_name=None,
                          verbose=True)
recorder.start()
time.sleep(10)
recorder.stop()
```
When the argument `record_dir` is set to None, the current folder obtained with
 `pathlib.Path.cwd()` is used. When the argument `fname` is set to None, the
 created files' stem use the start datetime.

**CLI:** The stream recorder can be called by command-line in a terminal by
using either `bsl stream_recorder` or `bsl_stream_recorder` followed by the
optional arguments `-d`, `-f`, `-s` respectively for `record_dir`, `fname`,
and `stream_name`, and the optional flags `--fif_subdir` and `--verbose`.
```
bsl_stream_recorder -d "D:/Data"
bsl_stream_recorder -d "D:/Data" -f test
bsl_stream_recorder -d "D:/Data" -f test -s openvibeSignals
```
## StreamPlayer
The stream player loads a previously recorded `.fif` file and creates a LSL
server streaming data from this file. The stream player can be used to test
code with a fake LSL data stream.

**Example:**
```
import time
from bsl import StreamPlayer

sp = StreamPlayer(stream_name='StreamPlayer', fif_file=r'path to .fif')
sp.start()
time.sleep(10)
sp.stop()
```
**CLI:**  The stream player can be called by command-line in a terminal by
using either `bsl stream_player` or `bsl_stream_player` followed by positional
arguments `stream_name` and `fif_file` and the optional arguments `-r`, `-c`,
`-t` respectively for `repeat`, `chunk_size` and `trigger_def`, and the
optional flag `--high_resolution`.
```
bsl_stream_player StreamPlayer data-raw.fif
bsl_stream_player StreamPlayer data-raw.fif -c 16
bsl_stream_player StreamPlayer data-raw.fif -c 16 -t triggerdef.ini
```
## StreamViewer
The stream viewer creates a 2-window GUI composed of a control GUI and a
plotter GUI to display the data acquired from an LSL server in real-time.

**CLI:** The stream viewer can be called by command-line in a terminal by using
either `bsl stream_viewer` or `bsl_stream_viewer` followed by the optional
argument `-s` for the `stream_name`. If no stream name is provided, a prompt
will ask the user to select the desired non-marker stream to display.
```
bsl_stream_viewer
bsl_stream_viewer -s StreamPlayer
```

## Triggers
Triggers includes functions to mark time event by sending a trigger which will
be saved on the *TRIGGER* channel of the on-going recording. Triggers can be
achieved either through hardware or through software.

Currently, the supported hardware triggers use an LPT port.

**Example:**
```
import time
from bsl import StreamRecorder
from bsl.triggers import SoftwareTrigger
from bsl.triggers import ParallelPortTrigger

# Software trigger
recorder = StreamRecorder()
recorder.start()
trigger = Softwaretrigger(recorder)
for k in range(1, 5):
    trigger.signal(k)
    time.sleep(1)
trigger.close()
recorder.stop()

# Hardware trigger through Arduino LPT converter
recorder = StreamRecorder()
recorder.start()
trigger = ParallelPortTrigger(address='arduino')
for k in range(1, 5):
    trigger.signal(k)
    time.sleep(1)
trigger.close()
recorder.stop()
```
Note that closing the trigger before stopping the recording may not be required
for all kind of triggers.

# Copyright and license
The codes are released under
[GNU Lesser General Public License](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html).
