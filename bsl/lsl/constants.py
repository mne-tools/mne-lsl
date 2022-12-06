import platform
import struct
from ctypes import (
    c_byte,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_longlong,
    c_short,
)

from .load_liblsl import lib

# -----------------
# Supported formats
# -----------------
# Value formats supported by LSL. LSL data streams are sequences of samples,
# each of which is a same-size vector of values with one of the below types.

# For up to 24-bit precision measurements in the appropriate physical unit (
# e.g., microvolts). Integers from -16777216 to 16777216 are represented
# accurately.
cf_float32 = 1
# For universal numeric data as long as permitted by network and disk budget.
# The largest representable integer is 53-bit.
cf_float64 = 2
# For variable-length ASCII strings or data blobs, such as video frames,
# complex event descriptions, etc.
cf_string = 3
# For high-rate digitized formats that require 32-bit precision. Depends
# critically on meta-data to represent meaningful units. Useful for
# application event codes or other coded data.
cf_int32 = 4
# For very high bandwidth signals or CD quality audio (for professional audio
# float is recommended).
cf_int16 = 5
# For binary signals or other coded data.
cf_int8 = 6
# For now only for future compatibility. Support for this type is not
# available on all languages and platforms.
cf_int64 = 7
# Can not be transmitted.
cf_undefined = 0

string2fmt = {
    "float32": cf_float32,
    "float64": cf_float64,
    "string": cf_string,
    "int8": cf_int8,
    "int16": cf_int16,
    "int32": cf_int32,
    "int64": cf_int64,
}

fmt2string = {value: key for key, value in string2fmt.items()}
fmt2type = [
    [],
    c_float,
    c_double,
    c_char_p,
    c_int,
    c_short,
    c_byte,
    c_longlong,
]

# ------------------------------
# Handle int64 incompatibilities
# ------------------------------
# int64 is not supported on windows and on 32 bits OS
if struct.calcsize("P") != 4 and platform.system() != "Windows":
    push_sample_int64 = lib.lsl_push_sample_ltp
    pull_sample_int64 = lib.lsl_pull_sample_l
    push_chunk_int64 = lib.lsl_push_chunk_ltp
    pull_chunk_int64 = lib.lsl_pull_chunk_l
else:

    def push_sample_int64(*_):  # noqa: D103
        raise NotImplementedError(
            "int64 is not yet supported on your platform."
        )

    pull_sample_int64 = push_chunk_int64 = pull_chunk_int64 = push_sample_int64

# --------------------
# Push/Pull properties
# --------------------

fmt2push_sample = [
    [],
    lib.lsl_push_sample_ftp,
    lib.lsl_push_sample_dtp,
    lib.lsl_push_sample_strtp,
    lib.lsl_push_sample_itp,
    lib.lsl_push_sample_stp,
    lib.lsl_push_sample_ctp,
    push_sample_int64,
]
fmt2pull_sample = [
    [],
    lib.lsl_pull_sample_f,
    lib.lsl_pull_sample_d,
    lib.lsl_pull_sample_str,
    lib.lsl_pull_sample_i,
    lib.lsl_pull_sample_s,
    lib.lsl_pull_sample_c,
    pull_sample_int64,
]
try:
    fmt2push_chunk = [
        [],
        lib.lsl_push_chunk_ftp,
        lib.lsl_push_chunk_dtp,
        lib.lsl_push_chunk_strtp,
        lib.lsl_push_chunk_itp,
        lib.lsl_push_chunk_stp,
        lib.lsl_push_chunk_ctp,
        push_chunk_int64,
    ]
    fmt2pull_chunk = [
        [],
        lib.lsl_pull_chunk_f,
        lib.lsl_pull_chunk_d,
        lib.lsl_pull_chunk_str,
        lib.lsl_pull_chunk_i,
        lib.lsl_pull_chunk_s,
        lib.lsl_pull_chunk_c,
        pull_chunk_int64,
    ]
except Exception:  # if not available
    fmt2push_chunk = [None, None, None, None, None, None, None, None]
    fmt2pull_chunk = [None, None, None, None, None, None, None, None]

# ---------------------
# Post processing flags
# ---------------------
# No automatic post-processing; return the ground-truth time stamps for manual
# post-processing.
proc_none = 0
# Perform automatic clock synchronization; equivalent to manually adding the
# time_correction().
proc_clocksync = 1
# Remove jitter from time stamps using a smoothing algorithm to the received
# time stamps.
proc_dejitter = 2
# Force the time-stamps to be monotonically ascending. Only makes sense if
# timestamps are dejittered.
proc_monotonize = 4
# Post-processing is thread-safe (same inlet can be read from by multiple
# threads).
proc_threadsafe = 8
# Bitwise OR operations
proc_ALL = (
    proc_none
    | proc_clocksync
    | proc_dejitter
    | proc_monotonize
    | proc_threadsafe
)
