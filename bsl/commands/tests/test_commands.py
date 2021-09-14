from subprocess import Popen, PIPE, STDOUT

import pytest

from bsl import __version__


commands = (
    'bsl',
    'bsl_stream_player', 'bsl stream_player',
    'bsl_stream_recorder', 'bsl stream_recorder',
    'bsl_stream_viewer', 'bsl stream_viewer')


@pytest.mark.parametrize('command', commands)
def test_help(command):
    """Test help display."""
    cmd = f'{command} --help'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    assert 'usage' in p.stdout.read().decode("utf-8").lower()


def test_version():
    cmd = 'bsl --version'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    assert __version__ in p.stdout.read().decode("utf-8").lower()
