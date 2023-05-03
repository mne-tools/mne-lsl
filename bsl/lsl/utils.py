from ctypes import POINTER, c_int, c_void_p, cast
from typing import Optional

from .load_liblsl import lib


# -- XML tree -----------------------------------------------------------------
class XMLElement:
    """A lightweight XML element tree modeling the .desc() field of StreamInfo.

    Has a name and can have multiple named children or have text content as
    value; attributes are omitted. Insider note: The interface is modeled after
    a subset of pugixml's node type and is compatible with it. See also
    http://pugixml.googlecode.com/svn/tags/latest/docs/manual/access.html for
    additional documentation.
    """

    def __init__(self, handle):
        """Construct a new XML element from existing handle."""
        self.e = c_void_p(handle)

    # -- Tree Navigation ------------------------------------------------------
    def first_child(self):
        """Get the first child of the element."""
        return XMLElement(lib.lsl_first_child(self.e))

    def last_child(self):
        """Get the last child of the element."""
        return XMLElement(lib.lsl_last_child(self.e))

    def child(self, name):
        """Get a child with a specified name."""
        return XMLElement(lib.lsl_child(self.e, str.encode(name)))

    def next_sibling(self, name=None):
        """Get the next sibling in the children list of the parent node.

        If a name is provided, the next sibling with the given name is
        returned.
        """
        if name is None:
            return XMLElement(lib.lsl_next_sibling(self.e))
        else:
            return XMLElement(lib.lsl_next_sibling_n(self.e, str.encode(name)))

    def previous_sibling(self, name=None):
        """Get the previous sibling in the children list of the parent node.

        If a name is provided, the previous sibling with the given name is
        returned.
        """
        if name is None:
            return XMLElement(lib.lsl_previous_sibling(self.e))
        else:
            return XMLElement(lib.lsl_previous_sibling_n(self.e, str.encode(name)))

    def parent(self):
        """Get the parent node."""
        return XMLElement(lib.lsl_parent(self.e))

    # -- Content Queries ------------------------------------------------------
    def empty(self):  # noqa: D401
        """True if this node is empty."""
        return bool(lib.lsl_empty(self.e))

    def is_text(self):  # noqa: D401
        """True if this node is a text body (instead of an XML element).

        True both for plain char data and CData.
        """
        return bool(lib.lsl_is_text(self.e))

    def name(self):
        """Name of the element."""
        return lib.lsl_name(self.e).decode("utf-8")

    def value(self):
        """Value of the element."""
        return lib.lsl_value(self.e).decode("utf-8")

    def child_value(self, name=None):
        """Get child value (value of the first child that is text).

        If a name is provided, then the value of the first child with the
        given name is returned.
        """
        if name is None:
            res = lib.lsl_child_value(self.e)
        else:
            res = lib.lsl_child_value_n(self.e, str.encode(name))
        return res.decode("utf-8")

    # -- Modification ---------------------------------------------------------
    def append_child_value(self, name, value):  # noqa: D205, D400
        """Append a child node with a given name, which has a (nameless)
        plain-text child with the given text value.
        """
        return XMLElement(
            lib.lsl_append_child_value(self.e, str.encode(name), str.encode(value))
        )

    def prepend_child_value(self, name, value):  # noqa: D205, D400
        """Prepend a child node with a given name, which has a (nameless)
        plain-text child with the given text value.
        """
        return XMLElement(
            lib.lsl_prepend_child_value(self.e, str.encode(name), str.encode(value))
        )

    def set_child_value(self, name, value):  # noqa: D205, D400
        """Set the text value of the (nameless) plain-text child of a named
        child node.
        """
        return XMLElement(
            lib.lsl_set_child_value(self.e, str.encode(name), str.encode(value))
        )

    def set_name(self, name):
        """Set the element's name.

        Return False if the node is empty.
        """
        return bool(lib.lsl_set_name(self.e, str.encode(name)))

    def set_value(self, value):
        """Set the element's value.

        Return False if the node is empty.
        """
        return bool(lib.lsl_set_value(self.e, str.encode(value)))

    def append_child(self, name):
        """Append a child element with the specified name."""
        return XMLElement(lib.lsl_append_child(self.e, str.encode(name)))

    def prepend_child(self, name):
        """Prepend a child element with the specified name."""
        return XMLElement(lib.lsl_prepend_child(self.e, str.encode(name)))

    def append_copy(self, elem):
        """Append a copy of the specified element as a child."""
        return XMLElement(lib.lsl_append_copy(self.e, elem.e))

    def prepend_copy(self, elem):
        """Prepend a copy of the specified element as a child."""
        return XMLElement(lib.lsl_prepend_copy(self.e, elem.e))

    def remove_child(self, rhs):
        """Remove a given child element, specified by name or as element."""
        if type(rhs) is XMLElement:
            lib.lsl_remove_child(self.e, rhs.e)
        else:
            lib.lsl_remove_child_n(self.e, rhs)


# -- Exception handling -------------------------------------------------------
class LostError(RuntimeError):  # noqa: D101
    pass


class InvalidArgumentError(RuntimeError):  # noqa: D101
    pass


class InternalError(RuntimeError):  # noqa: D101
    pass


def handle_error(errcode):
    """Error handler function.

    Translates an error code into an exception.
    """
    if type(errcode) is c_int:
        errcode = errcode.value
    if errcode == 0:
        pass  # no error
    elif errcode == -1:
        raise TimeoutError("The operation failed due to a timeout.")
    elif errcode == -2:
        raise LostError("The stream connection has been lost.")
    elif errcode == -3:
        raise InvalidArgumentError("An argument was incorrectly specified.")
    elif errcode == -4:
        raise InternalError("An internal error has occurred.")
    elif errcode < 0:
        raise RuntimeError("An unknown error has occurred.")


# -- Memory function ----------------------------------------------------------
def _free_char_p_array_memory(char_p_array):
    num_elements = len(char_p_array)
    pointers = cast(char_p_array, POINTER(c_void_p))
    for p in range(num_elements):
        if pointers[p] is not None:  # only free initialized pointers
            lib.lsl_destroy_string(pointers[p])


# -- Static checker -----------------------------------------------------------
def _check_timeout(timeout: Optional[float]) -> float:
    """Check that the provided timeout is valid.

    Parameters
    ----------
    timeout : float | None
        Timeout (in seconds) or None to disable timeout.

    Returns
    -------
    timeout : float
        Timeout (in seconds). If None was provided, a very large float is
        provided.
    """
    # with _check_type, the execution takes 800-900 ns.
    # with the try/except below, the execution takes 110 ns.
    if timeout is None:
        return 32000000.0  # about 1 year
    try:
        raise_ = timeout < 0
    except Exception:
        raise TypeError("The argument 'timeout' must be a strictly positive number.")
    if raise_:
        raise ValueError(
            "The argument 'timeout' must be a strictly positive number. "
            f"{timeout} is invalid."
        )
    return timeout
