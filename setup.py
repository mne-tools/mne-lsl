import os
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.dist import Distribution

_PATTERNS: dict[str, str] = {
    "Linux": "liblsl*.so*",
    "Darwin": "liblsl*.dylib",
    "Windows": "lsl*.dll",
}


class BinaryDistribution(Distribution):  # noqa: D101
    def has_ext_modules(self):  # noqa: D102
        return True


class build_ext(_build_ext):  # noqa: D101
    def run(self):
        """Build 'liblsl' with cmake as part of the extension build process."""
        skip = os.environ.get("MNE_LSL_SKIP_LIBLSL_BUILD")
        skip = eval(skip) if skip is not None else False
        if skip:
            print("Skipping build of liblsl.")  # noqa: T201
            return
        src = Path(__file__).parent / "src" / "liblsl"
        assert src.exists()  # sanity-check
        with TemporaryDirectory() as build_dir:
            args = [
                "cmake",
                "-S",
                str(src),
                "-B",
                build_dir,
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            if platform.system() == "Darwin":
                args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=11")
                args.append("-DLSL_FRAMEWORK=OFF")
            elif platform.system() == "Windows":
                args.extend(["-T", "v142,host=x64"])  # use VS2019 toolset
            unit_tests = os.environ.get("MNE_LSL_LIBLSL_BUILD_UNITTESTS")
            unit_tests = eval(unit_tests) if unit_tests is not None else False
            if unit_tests:
                args.append("-DLSL_UNITTESTS=ON")
            subprocess.run(args, check=True)
            subprocess.run(
                ["cmake", "--build", build_dir, "--config", "Release"], check=True
            )
            # locate the build files and move them to mne_lsl.lsl.lib
            if platform.system() == "Windows":
                build_dir = Path(build_dir) / "Release"
            else:
                build_dir = Path(build_dir)
            lib_files = [
                elt
                for elt in build_dir.glob(_PATTERNS[platform.system()])
                if not elt.is_symlink()
            ]
            assert len(lib_files) == 1  # sanity-check
            dst = (
                Path(__file__).parent / "src" / "mne_lsl" / "lsl" / "lib"
                if self.inplace
                else Path(self.build_lib) / "mne_lsl" / "lsl" / "lib"
            )
            dst.mkdir(parents=True, exist_ok=True)
            print(f"Copying {lib_files[0]} to {dst / lib_files[0].name}")  # noqa: T201
            shutil.copyfile(lib_files[0], dst / lib_files[0].name)
            # move unit test files if they were produced
            if unit_tests:
                if platform.system() == "Windows":
                    test_build_dir = build_dir.parent / "testing" / "Release"
                else:
                    test_build_dir = build_dir / "testing"
                test_files = [elt for elt in test_build_dir.glob("lsl_test*")]
                if len(test_files) == 0:
                    raise RuntimeError(
                        "The LIBLSL unit tests were requested but not found in the "
                        "build artifacts."
                    )
                dst = Path(__file__).parent / "tests" / "liblsl"
                dst.mkdir(parents=True, exist_ok=True)
                for test_file in test_files:
                    print(f"Moving {test_file} to {dst / test_file.name}")  # noqa: T201
                    shutil.move(test_file, dst / test_file.name)
                # also copy the liblsl files in the test directory
                for lib_file in build_dir.glob(_PATTERNS[platform.system()]):
                    print(f"Copying {lib_file} to {dst / lib_file.name}")  # noqa: T201
                    shutil.copyfile(
                        lib_file, dst / lib_file.name, follow_symlinks=False
                    )
        super().run()


class develop(_develop):  # noqa: D101
    def run(self):  # noqa: D102
        self.run_command("build_ext")
        super().run()


class bdist_wheel_abi3(bdist_wheel):  # noqa: D101
    def get_tag(self):  # noqa: D102
        python, abi, plat = super().get_tag()
        if python.startswith("cp"):
            return "cp310", "abi3", plat
        return python, abi, plat


setup(
    cmdclass={
        "build_ext": build_ext,
        "bdist_wheel": bdist_wheel_abi3,
        "develop": develop,
    },
    distclass=BinaryDistribution,
)
