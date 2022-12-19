import platform
import struct
from ctypes import (
    byref,
    c_byte,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_long,
    c_longlong,
    c_short,
    c_void_p,
)

from .load_liblsl import lib
from .utils import handle_error, _free_char_p_array_memory, XMLElement

# =================
# === Constants ===
# =================

# Value formats supported by LSL. LSL data streams are sequences of samples,
# each of which is a same-size vector of values with one of the below types.
# For up to 24-bit precision measurements in the appropriate physical unit (
# e.g., microvolts). Integers from -16777216 to 16777216 are represented
# accurately.
cf_float32 = 1
# For universal numeric data as long as permitted by network and disk budget.
#  The largest representable integer is 53-bit.
cf_double64 = 2
# For variable-length ASCII strings or data blobs, such as video frames,
# complex event descriptions, etc.
cf_string = 3
# For high-rate digitized formats that require 32-bit precision. Depends
# critically on meta-data to represent meaningful units. Useful for
# application event codes or other coded data.
cf_int32 = 4
# For very high bandwidth signals or CD quality audio (for professional audio
#  float is recommended).
cf_int16 = 5
# For binary signals or other coded data.
cf_int8 = 6
# For now only for future compatibility. Support for this type is not
# available on all languages and platforms.
cf_int64 = 7

# Post processing flags
proc_none = 0  # No automatic post-processing; return the ground-truth time stamps for manual post-processing.
proc_clocksync = 1  # Perform automatic clock synchronization; equivalent to manually adding the time_correction().
proc_dejitter = 2  # Remove jitter from time stamps using a smoothing algorithm to the received time stamps.
proc_monotonize = 4  # Force the time-stamps to be monotonically ascending. Only makes sense if timestamps are dejittered.
proc_threadsafe = 8  # Post-processing is thread-safe (same inlet can be read from by multiple threads).
proc_ALL = (
    proc_none
    | proc_clocksync
    | proc_dejitter
    | proc_monotonize
    | proc_threadsafe
)

# ==========================
# === Stream Declaration ===
# ==========================
class StreamInfo:
    def __init__(
        self,
        name="untitled",
        type="",
        channel_count=1,
        nominal_srate=0.0,
        channel_format=cf_float32,
        source_id="",
        handle=None,
    ):
        if handle is not None:
            self.obj = c_void_p(handle)
        else:
            if isinstance(channel_format, str):
                channel_format = string2fmt[channel_format]
            self.obj = lib.lsl_create_streaminfo(
                c_char_p(str.encode(name)),
                c_char_p(str.encode(type)),
                channel_count,
                c_double(nominal_srate),
                channel_format,
                c_char_p(str.encode(source_id)),
            )
            self.obj = c_void_p(self.obj)
            if not self.obj:
                raise RuntimeError(
                    "could not create stream description " "object."
                )

    def __del__(self):
        try:
            lib.lsl_destroy_streaminfo(self.obj)
        except:
            pass

    # === Core Information (assigned at construction) ===
    def name(self):
        return lib.lsl_get_name(self.obj).decode("utf-8")

    def type(self):
        return lib.lsl_get_type(self.obj).decode("utf-8")

    def channel_count(self):
        return lib.lsl_get_channel_count(self.obj)

    def nominal_srate(self):
        return lib.lsl_get_nominal_srate(self.obj)

    def channel_format(self):
        return lib.lsl_get_channel_format(self.obj)

    def source_id(self):
        return lib.lsl_get_source_id(self.obj).decode("utf-8")

    # === Hosting Information (assigned when bound to an outlet/inlet) ===
    def version(self):
        return lib.lsl_get_version(self.obj)

    def created_at(self):
        return lib.lsl_get_created_at(self.obj)

    def uid(self):
        return lib.lsl_get_uid(self.obj).decode("utf-8")

    def session_id(self):
        return lib.lsl_get_session_id(self.obj).decode("utf-8")

    def hostname(self):
        return lib.lsl_get_hostname(self.obj).decode("utf-8")

    # === Data Description (can be modified) ===
    def desc(self):
        return XMLElement(lib.lsl_get_desc(self.obj))

    def as_xml(self):
        return lib.lsl_get_xml(self.obj).decode("utf-8")


# =====================
# === Stream Outlet ===
# =====================


class StreamOutlet:
    def __init__(self, info, chunk_size=0, max_buffered=360):
        self.obj = lib.lsl_create_outlet(info.obj, chunk_size, max_buffered)
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("could not create stream outlet.")
        self.channel_format = info.channel_format()
        self.channel_count = info.channel_count()
        self.do_push_sample = fmt2push_sample[self.channel_format]
        self.do_push_chunk = fmt2push_chunk[self.channel_format]
        self.value_type = fmt2type[self.channel_format]
        self.sample_type = self.value_type * self.channel_count

    def __del__(self):
        try:
            lib.lsl_destroy_outlet(self.obj)
        except:
            pass

    def push_sample(self, x, timestamp=0.0, pushthrough=True):
        if len(x) == self.channel_count:
            if self.channel_format == cf_string:
                x = [v.encode("utf-8") for v in x]
            handle_error(
                self.do_push_sample(
                    self.obj,
                    self.sample_type(*x),
                    c_double(timestamp),
                    c_int(pushthrough),
                )
            )
        else:
            raise ValueError(
                "length of the sample (" + str(len(x)) + ") must "
                "correspond to the stream's channel count ("
                + str(self.channel_count)
                + ")."
            )

    def push_chunk(self, x, timestamp=0.0, pushthrough=True):
        try:
            n_values = self.channel_count * len(x)
            data_buff = (self.value_type * n_values).from_buffer(x)
            handle_error(
                self.do_push_chunk(
                    self.obj,
                    data_buff,
                    c_long(n_values),
                    c_double(timestamp),
                    c_int(pushthrough),
                )
            )
        except TypeError:
            if len(x):
                if type(x[0]) is list:
                    x = [v for sample in x for v in sample]
                if self.channel_format == cf_string:
                    x = [v.encode("utf-8") for v in x]
                if len(x) % self.channel_count == 0:
                    constructor = self.value_type * len(x)
                    # noinspection PyCallingNonCallable
                    handle_error(
                        self.do_push_chunk(
                            self.obj,
                            constructor(*x),
                            c_long(len(x)),
                            c_double(timestamp),
                            c_int(pushthrough),
                        )
                    )
                else:
                    raise ValueError(
                        "Each sample must have the same number of channels ("
                        + str(self.channel_count)
                        + ")."
                    )

    def have_consumers(self):
        return bool(lib.lsl_have_consumers(self.obj))

    def wait_for_consumers(self, timeout):
        return bool(lib.lsl_wait_for_consumers(self.obj, c_double(timeout)))

    def get_info(self):
        outlet_info = lib.lsl_get_info(self.obj)
        return StreamInfo(handle=outlet_info)


# =========================
# === Resolve Functions ===
# =========================


def resolve_streams(wait_time=1.0):
    buffer = (c_void_p * 1024)()
    num_found = lib.lsl_resolve_all(byref(buffer), 1024, c_double(wait_time))
    return [StreamInfo(handle=buffer[k]) for k in range(num_found)]


# ====================
# === Stream Inlet ===
# ====================


class StreamInlet:
    def __init__(
        self,
        info,
        max_buflen=360,
        max_chunklen=0,
        recover=True,
        processing_flags=0,
    ):
        if type(info) is list:
            raise TypeError(
                "description needs to be of type StreamInfo, " "got a list."
            )
        self.obj = lib.lsl_create_inlet(
            info.obj, max_buflen, max_chunklen, recover
        )
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("could not create stream inlet.")
        if processing_flags > 0:
            handle_error(
                lib.lsl_set_postprocessing(self.obj, processing_flags)
            )
        self.channel_format = info.channel_format()
        self.channel_count = info.channel_count()
        self.do_pull_sample = fmt2pull_sample[self.channel_format]
        self.do_pull_chunk = fmt2pull_chunk[self.channel_format]
        self.value_type = fmt2type[self.channel_format]
        self.sample_type = self.value_type * self.channel_count
        self.sample = self.sample_type()
        self.buffers = {}

    def __del__(self):
        try:
            lib.lsl_destroy_inlet(self.obj)
        except:
            pass

    def info(self, timeout=32000000.0):
        errcode = c_int()
        result = lib.lsl_get_fullinfo(
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return StreamInfo(handle=result)

    def open_stream(self, timeout=32000000.0):
        errcode = c_int()
        lib.lsl_open_stream(self.obj, c_double(timeout), byref(errcode))
        handle_error(errcode)

    def close_stream(self):
        lib.lsl_close_stream(self.obj)

    def time_correction(self, timeout=32000000.0):
        errcode = c_int()
        result = lib.lsl_time_correction(
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return result

    def pull_sample(self, timeout=32000000.0, sample=None):
        # support for the legacy API
        if type(timeout) is list:
            assign_to = timeout
            timeout = sample if type(sample) is float else 0.0
        else:
            assign_to = None

        errcode = c_int()
        timestamp = self.do_pull_sample(
            self.obj,
            byref(self.sample),
            self.channel_count,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)
        if timestamp:
            sample = [v for v in self.sample]
            if self.channel_format == cf_string:
                sample = [v.decode("utf-8") for v in sample]
            if assign_to is not None:
                assign_to[:] = sample
            return sample, timestamp
        else:
            return None, None

    def pull_chunk(self, timeout=0.0, max_samples=1024, dest_obj=None):
        num_channels = self.channel_count
        max_values = max_samples * num_channels

        if max_samples not in self.buffers:
            # noinspection PyCallingNonCallable
            self.buffers[max_samples] = (
                (self.value_type * max_values)(),
                (c_double * max_samples)(),
            )
        if dest_obj is not None:
            data_buff = (self.value_type * max_values).from_buffer(dest_obj)
        else:
            data_buff = self.buffers[max_samples][0]
        ts_buff = self.buffers[max_samples][1]

        # read data into it
        errcode = c_int()
        num_elements = self.do_pull_chunk(
            self.obj,
            byref(data_buff),
            byref(ts_buff),
            max_values,
            max_samples,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)
        # return results (note: could offer a more efficient format in the
        # future, e.g., a numpy array)
        num_samples = num_elements / num_channels
        if dest_obj is None:
            samples = [
                [data_buff[s * num_channels + c] for c in range(num_channels)]
                for s in range(int(num_samples))
            ]
            if self.channel_format == cf_string:
                samples = [[v.decode("utf-8") for v in s] for s in samples]
                _free_char_p_array_memory(data_buff, max_values)
        else:
            samples = None
        timestamps = [ts_buff[s] for s in range(int(num_samples))]
        return samples, timestamps

    def samples_available(self):
        return lib.lsl_samples_available(self.obj)

    def flush(self):
        return lib.lsl_inlet_flush(self.obj)

    def was_clock_reset(self):
        return bool(lib.lsl_was_clock_reset(self.obj))


# ==================================
# === Module Initialization Code ===
# ==================================
# int64 support on windows and 32bit OSes isn't there yet
if struct.calcsize("P") != 4 and platform.system() != "Windows":
    push_sample_int64 = lib.lsl_push_sample_ltp
    pull_sample_int64 = lib.lsl_pull_sample_l
    push_chunk_int64 = lib.lsl_push_chunk_ltp
    pull_chunk_int64 = lib.lsl_pull_chunk_l
else:

    def push_sample_int64(*_):
        raise NotImplementedError(
            "int64 support isn't enabled on your platform"
        )

    pull_sample_int64 = push_chunk_int64 = pull_chunk_int64 = push_sample_int64

# set up some type maps
string2fmt = {
    "float32": cf_float32,
    "double64": cf_double64,
    "string": cf_string,
    "int32": cf_int32,
    "int16": cf_int16,
    "int8": cf_int8,
    "int64": cf_int64,
}
fmt2string = [
    "undefined",
    "float32",
    "double64",
    "string",
    "int32",
    "int16",
    "int8",
    "int64",
]
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
