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
