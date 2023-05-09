import platform
import struct
from ctypes import c_byte, c_char_p, c_double, c_float, c_int, c_longlong, c_short

import numpy as np

from .load_liblsl import lib

# -----------------
# Supported formats
# -----------------
# Value formats supported by LSL. LSL data streams are sequences of samples, each of
# which is a same-size vector of values with one of the below types.

string2fmt = {
    "float32": c_float,
    "float64": c_double,
    "string": c_char_p,
    "int8": c_byte,
    "int16": c_short,
    "int32": c_int,
    "int64": c_longlong,
}
fmt2string = {value: key for key, value in string2fmt.items()}

idx2fmt = {
    1: c_float,
    2: c_double,
    3: c_char_p,
    4: c_int,
    5: c_short,
    6: c_byte,
    7: c_longlong,
}
fmt2idx = {value: key for key, value in idx2fmt.items()}

numpy2fmt = {
    np.float32: c_float,
    np.float64: c_double,
    np.int8: c_byte,
    np.int16: c_short,
    np.int32: c_int,
    np.int64: c_longlong,
}
fmt2numpy = {value: key for key, value in numpy2fmt.items()}

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
        raise NotImplementedError("int64 is not yet supported on your platform.")

    pull_sample_int64 = push_chunk_int64 = pull_chunk_int64 = push_sample_int64

# -------------------
# Push/Pull functions
# -------------------

fmt2push_sample = {
    c_float: lib.lsl_push_sample_ftp,
    c_double: lib.lsl_push_sample_dtp,
    c_char_p: lib.lsl_push_sample_strtp,
    c_int: lib.lsl_push_sample_itp,
    c_short: lib.lsl_push_sample_stp,
    c_byte: lib.lsl_push_sample_ctp,
    c_longlong: push_sample_int64,
}
fmt2pull_sample = {
    c_float: lib.lsl_pull_sample_f,
    c_double: lib.lsl_pull_sample_d,
    c_char_p: lib.lsl_pull_sample_str,
    c_int: lib.lsl_pull_sample_i,
    c_short: lib.lsl_pull_sample_s,
    c_byte: lib.lsl_pull_sample_c,
    c_longlong: pull_sample_int64,
}
fmt2push_chunk = {
    c_float: lib.lsl_push_chunk_ftp,
    c_double: lib.lsl_push_chunk_dtp,
    c_char_p: lib.lsl_push_chunk_strtp,
    c_int: lib.lsl_push_chunk_itp,
    c_short: lib.lsl_push_chunk_stp,
    c_byte: lib.lsl_push_chunk_ctp,
    c_longlong: push_chunk_int64,
}
fmt2pull_chunk = {
    c_float: lib.lsl_pull_chunk_f,
    c_double: lib.lsl_pull_chunk_d,
    c_char_p: lib.lsl_pull_chunk_str,
    c_int: lib.lsl_pull_chunk_i,
    c_short: lib.lsl_pull_chunk_s,
    c_byte: lib.lsl_pull_chunk_c,
    c_longlong: pull_chunk_int64,
}

# ---------------------
# Post processing flags
# ---------------------
post_processing_flags = {
    "clocksync": 1,
    "dejitter": 2,
    "monotize": 4,
    "threadsafe": 8,
}
