import mne_lsl
import mne_lsl._ffi_liblsl

mne_lsl.sys_info()
print(f"{mne_lsl._ffi_liblsl.lib.lsl_protocol_version()=}")  # noqa: T201
