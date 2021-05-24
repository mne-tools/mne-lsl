'''
Trigger events sending

Triggers are used to mark event (stimulus) timings during a recording.

It supports the following types:

- Desktop native LPT ('DESKTOP')
- Commercial USB2LPT adapter ('USB2LPT')
- Software trigger ('SOFTWARE')
- Arduino trigger box ('ARDUINO')
- Mock trigger device for testing ('FAKE')

The asynchronous function signal(x) sends 1-byte integer value x and returns immediately.
It schedules to send the value 0 at the end of the signal length, defined by init().

**For USB2LPT:** download the driver from here https://www-user.tu-chemnitz.de/~heha/bastelecke/Rund%20um%20den%20PC/USB2LPT/

**For SOFTWARE:** write events information into a text file with LSL timestamps, which
can be later added to fif. This file will be automatically saved and closed when Ctrl+C is
pressed or terminal window is closed (or killed for whatever reason).

**For Arduino:** go here https://github.com/fcbg-hnp/arduino-trigger to build the Arduino converter.
'''

from .trigger import Trigger
from .trigger_def import TriggerDef
