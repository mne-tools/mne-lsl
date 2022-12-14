"""Utility functions for checking types and values. Inspired from MNE."""

import logging
import operator
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._docs import fill_doc


def _ensure_int(item: Any, item_name: Optional[str] = None) -> int:
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
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(item, bool):
            raise TypeError
        item = int(operator.index(item))
    except TypeError:
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError(
            "%s must be an int, got %s instead." % (item_name, type(item))
        )

    return item


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool:
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool:
        return callable(other)


_types = {
    "numeric": (np.floating, float, _IntLike()),
    "path-like": (str, Path, os.PathLike),
    "int": (_IntLike(),),
    "callable": (_Callable(),),
}


def _check_type(
    item: Any, types: tuple, item_name: Optional[str] = None
) -> Any:
    """Check that item is an instance of types.

    Parameters
    ----------
    item : object
        Item to check.
    types : tuple of types | tuple of str
        Types to be checked against.
        If str, must be one of:
            ('int', 'str', 'numeric', 'path-like', 'callable')
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """
    check_types = sum(
        (
            (type(None),)
            if type_ is None
            else (type_,)
            if not isinstance(type_, str)
            else _types[type_]
            for type_ in types
        ),
        (),
    )

    if not isinstance(item, check_types):
        type_name = [
            "None"
            if cls_ is None
            else cls_.__name__
            if not isinstance(cls_, str)
            else cls_
            for cls_ in types
        ]
        if len(type_name) == 1:
            type_name = type_name[0]
        elif len(type_name) == 2:
            type_name = " or ".join(type_name)
        else:
            type_name[-1] = "or " + type_name[-1]
            type_name = ", ".join(type_name)
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError(
            f"{item_name} must be an instance of {type_name}, "
            f"got {type(item)} instead."
        )

    return item


def _check_value(
    item: Any,
    allowed_values: tuple,
    item_name: Optional[str] = None,
    extra: Optional[str] = None,
) -> Any:
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
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """
    if item not in allowed_values:
        item_name = "" if item_name is None else " '%s'" % item_name
        extra = "" if extra is None else " " + extra
        msg = (
            "Invalid value for the{item_name} parameter{extra}. "
            "{options}, but got {item!r} instead."
        )
        allowed_values = tuple(allowed_values)  # e.g., if a dict was given
        if len(allowed_values) == 1:
            options = "The only allowed value is %s" % repr(allowed_values[0])
        elif len(allowed_values) == 2:
            options = "Allowed values are %s and %s" % (
                repr(allowed_values[0]),
                repr(allowed_values[1]),
            )
        else:
            options = "Allowed values are "
            options += ", ".join([f"{repr(v)}" for v in allowed_values[:-1]])
            options += f", and {repr(allowed_values[-1])}"
        raise ValueError(
            msg.format(
                item_name=item_name, extra=extra, options=options, item=item
            )
        )

    return item


@fill_doc
def _check_verbose(verbose: Any) -> int:
    """Check that the value of verbose is valid.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    verbose : int
        The verbosity level as an integer.
    """
    logging_types = dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARNING=logging.WARNING,
        ERROR=logging.ERROR,
        CRITICAL=logging.CRITICAL,
    )

    _check_type(verbose, (bool, str, "int", None), item_name="verbose")

    if verbose is None:
        verbose = logging.WARNING
    elif isinstance(verbose, str):
        verbose = verbose.upper()
        _check_value(verbose, logging_types, item_name="verbose")
        verbose = logging_types[verbose]
    elif isinstance(verbose, bool):
        if verbose:
            verbose = logging.INFO
        else:
            verbose = logging.WARNING
    elif isinstance(verbose, int):
        if verbose <= 0:
            raise ValueError(
                "Argument 'verbose' can not be a negative integer, "
                f"{verbose} is invalid."
            )

    return verbose
