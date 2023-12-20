from pathlib import Path
from typing import Any, Optional

from _typeshed import Incomplete

from ._docs import fill_doc as fill_doc

def ensure_int(item: Any, item_name: Optional[str] = ...) -> int:
    """Ensure a variable is an integer.

    Parameters
    ----------
    item : Any
        Item to check.
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not int.
    """

class _IntLike:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool: ...

class _Callable:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool: ...

_types: Incomplete

def check_type(item: Any, types: tuple, item_name: Optional[str] = ...) -> None:
    """Check that item is an instance of types.

    Parameters
    ----------
    item : object
        Item to check.
    types : tuple of types | tuple of str
        Types to be checked against.
        If str, must be one of ('int-like', 'numeric', 'path-like', 'callable',
        'array-like').
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """

def check_value(
    item: Any,
    allowed_values: tuple,
    item_name: Optional[str] = ...,
    extra: Optional[str] = ...,
) -> None:
    """Check the value of a parameter against a list of valid options.

    Parameters
    ----------
    item : object
        Item to check.
    allowed_values : tuple of objects
        Allowed values to be checked against.
    item_name : str | None
        Name of the item to show inside the error message.
    extra : str | None
        Extra string to append to the invalid value sentence, e.g. "when using DC mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """

def check_verbose(verbose: Any) -> int:
    """Check that the value of verbose is valid.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between
        ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
        If None is provided, the verbosity is set to ``"WARNING"``.
        If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
        to ``"INFO"`` for True.

    Returns
    -------
    verbose : int
        The verbosity level as an integer.
    """

def ensure_path(item: Any, must_exist: bool) -> Path:
    """Ensure a variable is a Path.

    Parameters
    ----------
    item : Any
        Item to check.
    must_exist : bool
        If True, the path must resolve to an existing file or directory.

    Returns
    -------
    path : Path
        Path validated and converted to a pathlib.Path object.
    """
