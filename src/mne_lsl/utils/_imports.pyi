from types import ModuleType

_INSTALL_MAPPING: dict[str, str]

def import_optional_dependency(
    name: str, extra: str = "", raise_error: bool = True, *, conda: bool = True
) -> ModuleType | None:
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    raise_error : bool
        What to do when a dependency is not found.
        * True : Raise an ImportError.
        * False: Return None.
    conda : bool
        If True, the ImportError message mentions both ``pip`` and ``conda`` as
        installation methods. If False, only ``pip`` is mentioned, e.g. for packages
        which are not distributed on ``conda-forge``.

    Returns
    -------
    module : Module | None
        The imported module when found.
        None is returned when the package is not found and raise_error is False.
    """
