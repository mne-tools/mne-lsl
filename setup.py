# Minimal setup.py to create a CFFI extension module in abi3 mode if requested

import os

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class bdist_wheel_abi3(bdist_wheel):  # noqa: D101
    def get_tag(self):  # noqa: D102
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            return "cp310", "abi3", plat

        return python, abi, plat


kwargs = dict()
if os.getenv("MNE_LSL_BUILD_FFI", "0").lower() in ("1", "true"):
    kwargs.update(
        cffi_modules=["tools/ffi_build.py:ffibuilder"],
        cmdclass={"bdist_wheel": bdist_wheel_abi3},
    )

if __name__ == "__main__":
    setup(**kwargs)
