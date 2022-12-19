from ctypes import (
    byref,
    c_char_p,
    c_double,
    c_int,
    c_long,
    c_void_p,
)

from ...utils._checks import _check_value, _check_type
from .load_liblsl import lib
from .utils import handle_error, _free_char_p_array_memory, XMLElement
from .constants import fmt2pull_sample, fmt2pull_chunk, fmt2idx, idx2fmt, string2fmt, fmt2push_sample, fmt2push_chunk


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
        channel_format="float32",
        source_id="",
        handle=None,
    ):
        if handle is not None:
            self.obj = c_void_p(handle)
        else:
            if isinstance(channel_format, str):
                channel_format = StreamInfo._string2idxfmt(channel_format)
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _string2idxfmt(dtype) -> int:
        """Convert a string format to its LSL integer value."""
        if dtype in fmt2idx:
            return fmt2idx[dtype]
        _check_type(dtype, (str, "int"), "dtype")
        if isinstance(dtype, str):
            _check_value(dtype, string2fmt, "dtype")
            dtype = fmt2idx[string2fmt[dtype]]
        else:
            _check_value(dtype, idx2fmt, "dtype")
        return dtype

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
        return idx2fmt[lib.lsl_get_channel_format(self.obj)]

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
        self.value_type = self.channel_format
        self.sample_type = self.value_type * self.channel_count

    def __del__(self):
        try:
            lib.lsl_destroy_outlet(self.obj)
        except:
            pass

    def push_sample(self, x, timestamp=0.0, pushthrough=True):
        if len(x) == self.channel_count:
            if self.channel_format == c_char_p:
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
                if self.channel_format == c_char_p:
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
        self.value_type = self.channel_format
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
            if self.channel_format == c_char_p:
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
            if self.channel_format == c_char_p:
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
