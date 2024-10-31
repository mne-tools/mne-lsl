# https://cffi.readthedocs.io/en/latest/overview.html#main-mode-of-usage

from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef("signed int lsl_protocol_version();")
ffibuilder.set_source("mne_lsl._ffi_liblsl", '#include "lsl_c.h"', libraries=["lsl"])
if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
