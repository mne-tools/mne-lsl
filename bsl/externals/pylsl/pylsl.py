from ctypes import (
    byref,
    c_char_p,
    c_double,
    c_int,
    c_void_p,
)

from .load_liblsl import lib
from .utils import handle_error, _free_char_p_array_memory
from .constants import fmt2pull_sample, fmt2pull_chunk, string2fmt
from .stream_info import _BaseStreamInfo


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
            info._obj, max_buflen, max_chunklen, recover
        )
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("could not create stream inlet.")
        if processing_flags > 0:
            handle_error(
                lib.lsl_set_postprocessing(self.obj, processing_flags)
            )
        self.channel_format = string2fmt[info.dtype]
        self.channel_count = info.n_channels
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
        return _BaseStreamInfo(result)

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
