from _typeshed import Incomplete

from .load_liblsl import lib as lib

class XMLElement:
    """A lightweight XML element tree modeling the .desc() field of StreamInfo.

    Has a name and can have multiple named children or have text content as value;
    attributes are omitted. Insider note: The interface is modeled after a subset of
    pugixml's node type and is compatible with it.
    """

    e: Incomplete

    def __init__(self, handle) -> None:
        """Construct a new XML element from existing handle."""

    def first_child(self):
        """Get the first child of the element."""

    def last_child(self):
        """Get the last child of the element."""

    def child(self, name):
        """Get a child with a specified name."""

    def next_sibling(self, name: Incomplete | None = None):
        """Get the next sibling in the children list of the parent node.

        If a name is provided, the next sibling with the given name is returned.
        """

    def previous_sibling(self, name: Incomplete | None = None):
        """Get the previous sibling in the children list of the parent node.

        If a name is provided, the previous sibling with the given name is returned.
        """

    def parent(self):
        """Get the parent node."""

    def empty(self):
        """True if this node is empty."""

    def is_text(self):
        """True if this node is a text body (instead of an XML element).

        True both for plain char data and CData.
        """

    def name(self):
        """Name of the element."""

    def value(self):
        """Value of the element."""

    def child_value(self, name: Incomplete | None = None):
        """Get child value (value of the first child that is text).

        If a name is provided, then the value of the first child with the given name is
        returned.
        """

    def append_child_value(self, name, value):
        """Append a child node with a given name, which has a (nameless) plain-text
        child with the given text value.
        """

    def prepend_child_value(self, name, value):
        """Prepend a child node with a given name, which has a (nameless) plain-text
        child with the given text value.
        """

    def set_child_value(self, name, value):
        """Set the text value of the (nameless) plain-text child of a named
        child node.
        """

    def set_name(self, name):
        """Set the element's name.

        Return False if the node is empty.
        """

    def set_value(self, value):
        """Set the element's value.

        Return False if the node is empty.
        """

    def append_child(self, name):
        """Append a child element with the specified name."""

    def prepend_child(self, name):
        """Prepend a child element with the specified name."""

    def append_copy(self, elem):
        """Append a copy of the specified element as a child."""

    def prepend_copy(self, elem):
        """Prepend a copy of the specified element as a child."""

    def remove_child(self, rhs) -> None:
        """Remove a given child element, specified by name or as element."""

class LostError(RuntimeError): ...
class InvalidArgumentError(RuntimeError): ...
class InternalError(RuntimeError): ...

def handle_error(errcode) -> None:
    """Error handler function.

    Translates an error code into an exception.
    """

def free_char_p_array_memory(char_p_array) -> None: ...
def check_timeout(timeout: float | None) -> float:
    """Check that the provided timeout is valid.

    Parameters
    ----------
    timeout : float | None
        Timeout (in seconds) or None to disable timeout.

    Returns
    -------
    timeout : float
        Timeout (in seconds). If None was provided, a very large float is returned.
    """
