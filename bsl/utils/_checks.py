"""Utility functions for checking types and values. Inspired from MNE."""

import os
import operator
from pathlib import Path

import numpy as np


def _ensure_int(x, name='unknown'):
    """
    Ensure a variable is an integer.

    Parameters
    ----------
    x : Variable to test.
    name : str
        Variable name displayed in the error.

    Returns
    -------
    x : Variable tested.
    """
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be an int, got %s' % (name, type(x)))
    return x


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable:
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_types = {
    'numeric': (np.floating, float, _IntLike()),
    'path-like': (str, Path, os.PathLike),
    'int': (_IntLike(), ),
    'callable': (_Callable(), ),
}


def _check_type(item, types=None, item_name=None, type_name=None):
    """
    Check that item is an instance of types.

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
    type_name : str | None
        Possible types to show inside the error message that the checked item
        can be.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """
    check_types = sum(((type(None), ) if type_ is None else (type_, )
                       if not isinstance(type_, str) else _types[type_]
                       for type_ in types), ())

    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         if not isinstance(cls_, str) else cls_
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        _item_name = 'Item' if item_name is None else item_name
        raise TypeError(f"{_item_name} must be an instance of {type_name}, "
                        f"got {type(item)} instead.")


def _check_value(item, allowed_values, item_name, extra=''):
    """
    Check the value of a parameter against a list of valid options.

    Parameters
    ----------
    item : object
        Item to check.
    allowed_values : tuple of objects
        Allowed values to be checked against.
    item_name : str
        Name of the item to show inside the error message.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """
    if item not in allowed_values:
        extra = ' ' + extra if extra else extra
        msg = ("Invalid value for the '{item_name}' parameter{extra}. "
               '{options}, but got {item!r} instead.')
        allowed_values = tuple(allowed_values)  # e.g., if a dict was given
        if len(allowed_values) == 1:
            options = 'The only allowed value is %s' % repr(allowed_values[0])
        elif len(allowed_values) == 2:
            options = 'Allowed values are %s and %s' % \
                (repr(allowed_values[0]), repr(allowed_values[1]))
        else:
            options = 'Allowed values are '
            options += ', '.join([f'{repr(v)}' for v in allowed_values[:-1]])
            options += f', and {repr(allowed_values[-1])}'
        raise ValueError(msg.format(item_name=item_name, options=options,
                                    item=item, extra=extra))
