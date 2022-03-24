"""Test _docs.py"""

from bsl.utils._docs import fill_doc, copy_doc


def test_fill_doc():
    """Test decorator to fill docstring."""

    @fill_doc
    def foo(stream_name):
        """My doc.

        Parameters
        ----------
        %(stream_name)s
        """
        pass

    assert 'stream_name : list | str | None' in foo.__doc__


def test_copy_doc():
    """Test decorator to copy docstring."""

    def foo(x, y):
        """
        My doc.
        """
        pass

    @copy_doc(foo)
    def foo2(x, y):
        pass

    assert 'My doc.' in foo2.__doc__
