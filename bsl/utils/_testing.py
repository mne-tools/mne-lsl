"""Utility function for testing. Inspired from MNE."""

import sys
import requests
from io import StringIO

import pytest


def requires_good_network(function):
    """Decorator to skip a test if a network connection is not available."""
    try:
        requests.get('https://github.com/', timeout=1)
        skip = False
    except ConnectionError:
        skip = True
    name = function.__name__
    reason = 'Test %s skipped, requires a good network connection.' % name
    return pytest.mark.skipif(skip, reason=reason)(function)


class ClosingStringIO(StringIO):
    """StringIO that closes after getvalue()."""

    def getvalue(self, close=True):
        """Get the value."""
        out = super().getvalue()
        if close:
            self.close()
        return out


class ArgvSetter:
    """Context manager to temporarily set sys.argv."""

    def __init__(self, args=(), disable_stdout=True, disable_stderr=True):
        self.argv = list(('python',) + args)
        self.stdout = ClosingStringIO() if disable_stdout else sys.stdout
        self.stderr = ClosingStringIO() if disable_stderr else sys.stderr

    def __enter__(self):
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
