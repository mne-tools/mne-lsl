import platform
import re
import sys
from functools import partial
from importlib import import_module
from importlib.metadata import requires
from typing import IO, Callable, List, Optional

import psutil

from ._checks import _check_type
from ._imports import INSTALL_MAPPING

LIB_MAPPING = {value: key for key, value in INSTALL_MAPPING.items()}


def sys_info(fid: Optional[IO] = None, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`.
        Can be None to use :data:`sys.stdout`.
    developer : bool
        If True, display information about optional dependencies.
    """
    _check_type(developer, (bool,), "developer")

    ljust = 18
    out = partial(print, end="", file=fid)
    package = __package__.split(".")[0]

    # OS information - requires python 3.8 or above
    out("Platform:".ljust(ljust) + platform.platform() + "\n")
    # Python information
    out("Python:".ljust(ljust) + sys.version.replace("\n", " ") + "\n")
    out("Executable:".ljust(ljust) + sys.executable + "\n")
    # CPU information
    out("CPU:".ljust(ljust) + platform.processor() + "\n")
    out("Physical cores:".ljust(ljust) + str(psutil.cpu_count(False)) + "\n")
    out("Logical cores:".ljust(ljust) + str(psutil.cpu_count(True)) + "\n")
    # Memory information
    out("RAM:".ljust(ljust))
    out(f"{psutil.virtual_memory().total / float(2 ** 30):0.1f} GB\n")
    out("SWAP:".ljust(ljust))
    out(f"{psutil.swap_memory().total / float(2 ** 30):0.1f} GB\n")

    # dependencies
    out("\nDependencies info\n")
    out(f"{package}:".ljust(ljust) + import_module(package).__version__ + "\n")
    dependencies = [elt for elt in requires(package) if "extra" not in elt]
    _list_dependencies_info(out, ljust, dependencies)

    # extras
    if developer:
        keys = (
            "build",
            "test",
            "style",
        )
        for key in keys:
            out(f"\nOptional '{key}' info\n")
            dependencies = [
                elt.split(";")[0].rstrip()
                for elt in requires(package)
                if "extra" in elt and key in elt
            ]
            _list_dependencies_info(out, ljust, dependencies)


def _list_dependencies_info(
    out: Callable, ljust: int, dependencies: List[str]
):
    """List dependencies names and versions."""
    for dep in dependencies:
        # handle dependencies with version specifiers
        specifiers_pattern = r"(~=|==|!=|<=|>=|<|>|===)"
        specifiers = re.findall(specifiers_pattern, dep)
        if len(specifiers) != 0:
            dep, _ = dep.split(specifiers[0])
            while not dep[-1].isalpha():
                dep = dep[:-1]
        # handle dependencies provided with a [key], e.g. pydocstyle[toml]
        if "[" in dep:
            dep = dep.split("[")[0]
        # handle dependencies with a different name between PyPI and the lib
        if dep in LIB_MAPPING:
            dep = LIB_MAPPING[dep]
        try:
            module = import_module(dep)
            version = module.__version__
        except Exception:
            version = "Not found."
        out(f"{dep}:".ljust(ljust) + version + "\n")
