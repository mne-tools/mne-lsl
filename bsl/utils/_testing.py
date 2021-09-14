"""Utility function for testing. Inspired from MNE."""

import os
from functools import partial
from contextlib import contextmanager

import pytest


@contextmanager
def modified_env(**kwargs):
    """
    Use a modified os.environ with temporarily replaced key/value pairs.

    Parameters
    ----------
    **kwargs : dict
        The key/value pairs of environment variables to replace.
    """
    orig_env = dict()
    for key, val in kwargs.items():
        orig_env[key] = os.getenv(key)
        if val is not None:
            if isinstance(val, str):
                os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]
    try:
        yield
    finally:
        for key, val in orig_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]


def requires_module(function, name, call=None):
    """Skip a test if package is not available (decorator)."""
    call = ('import %s' % name) if call is None else call
    reason = 'Test %s skipped, requires %s.' % (function.__name__, name)
    try:
        exec(call) in globals(), locals()
    except Exception as exc:
        if len(str(exc)) > 0 and str(exc) != 'No module named %s' % name:
            reason += ' Got exception (%s)' % (exc,)
        skip = True
    else:
        skip = False
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_good_network = partial(
    requires_module, name='good network connection',
    call='if int(os.environ.get("BSL_SKIP_NETWORK_TESTS", 0)):\n'
         '    raise ImportError')
