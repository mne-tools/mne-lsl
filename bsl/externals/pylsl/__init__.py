# (not using import * for Python 2.5 support)
from .pylsl import IRREGULAR_RATE, DEDUCED_TIMESTAMP, FOREVER, cf_float32,\
    cf_double64, cf_string, cf_int32, cf_int16, cf_int8, cf_int64,\
    cf_undefined, protocol_version, library_version, local_clock,\
    proc_ALL, proc_none, proc_clocksync, proc_dejitter, proc_monotonize, proc_threadsafe,\
    StreamInfo, StreamOutlet, resolve_streams, \
    StreamInlet, XMLElement,\
    TimeoutError, LostError, InvalidArgumentError, InternalError

from .version import __version__
