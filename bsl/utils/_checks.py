import os
import operator
from pathlib import Path

import numpy as np


def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be %s, got %s' % (name, must_be, type(x)))
    return x


class _IntLike(object):
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable(object):
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_multi = {
    'str': (str, ),
    'numeric': (np.floating, float, _IntLike()),
    'path-like': (str, Path, os.PathLike),
    'int-like': (_IntLike(), ),
    'callable': (_Callable(), ),
}


def _check_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.

    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'int-like', 'str', 'numeric', 'info',
         'path-like', 'callable'}.
         If a tuple of str is passed, use 'int-like' and not 'int' for
         integers.
    item_name : str | None
        Name of the item to show inside the error message.
    type_name : str | None
        Possible types to show inside the error message that the checked item
        can be.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "info":
        from mne.io import Info as types

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(((type(None),) if type_ is None else (type_,)
                       if not isinstance(type_, str) else _multi[type_]
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


def _check_value(parameter, value, allowed_values, extra=''):
    """Check the value of a parameter against a list of valid options.

    Return the value if it is valid, otherwise raise a ValueError with a
    readable error message.

    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the parameter is not one of the valid options.

    Returns
    -------
    value : any type
        The value if it is valid.
    """
    if value in allowed_values:
        return value

    # Prepare a nice error message for the user
    extra = ' ' + extra if extra else extra
    msg = ("Invalid value for the '{parameter}' parameter{extra}. "
           '{options}, but got {value!r} instead.')
    allowed_values = list(allowed_values)  # e.g., if a dict was given
    if len(allowed_values) == 1:
        options = f'The only allowed value is {repr(allowed_values[0])}'
    else:
        options = 'Allowed values are '
        options += ', '.join([f'{repr(v)}' for v in allowed_values[:-1]])
        options += f', and {repr(allowed_values[-1])}'
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value, extra=extra))
