from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import Transform
from mne.io import read_raw_fif
from mne.utils import check_version

from mne_lsl.datasets import testing
from mne_lsl.lsl import StreamInfo, StreamOutlet
from mne_lsl.stream import StreamLSL
from mne_lsl.stream._hpi import CH_NAMES, check_hpi_ch_names

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from mne.io import BaseRaw


@pytest.fixture(scope="module")
def raw_without_dev_head_t() -> BaseRaw:
    """Fixture to provide a raw object without dev_head_t."""
    raw = read_raw_fif(testing.data_path() / "mne-sample" / "sample_audvis_raw.fif")
    with raw.info._unlock(update_redundant=False, check_after=False):
        raw.info["dev_head_t"] = None
    return raw


class DummyPlayer:
    """Dummy player object containing the player attributes."""

    def __init__(self, /, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _player_mock_lsl_stream(
    raw_without_dev_head_t: BaseRaw,
    chunk_size: int,
    name: str,
    source_id: str,
    status: mp.managers.ValueProxy,
) -> None:
    """Player for the 'mock_lsl_stream' fixture."""
    # nest the PlayerLSL import to first write the temporary LSL configuration file
    from mne_lsl.player import PlayerLSL  # noqa: E402

    player = PlayerLSL(
        raw_without_dev_head_t, chunk_size=chunk_size, name=name, source_id=source_id
    )
    player.start()
    status.value = 1
    while status.value:
        time.sleep(0.1)
    player.stop()


@pytest.fixture
def mock_megin_lsl_stream(
    raw_without_dev_head_t: BaseRaw, request: pytest.FixtureRequest, chunk_size: int
) -> Generator[DummyPlayer, None, None]:
    """Create a mock LSL stream for testing."""
    manager = mp.Manager()
    status = manager.Value("i", 0)
    name = f"P_{request.node.name}"
    source_id = uuid.uuid4().hex
    process = mp.Process(
        target=_player_mock_lsl_stream,
        args=(raw_without_dev_head_t, chunk_size, name, source_id, status),
    )
    process.start()
    while status.value != 1:
        pass
    yield DummyPlayer(name=name, source_id=source_id)
    status.value = 0
    process.join(timeout=2)
    process.kill()


@pytest.fixture(scope="module")
def hpi_stream_sinfo() -> StreamInfo:
    """Create a StreamInfo object for a mock HPI stream."""
    source_id = uuid.uuid4().hex
    sinfo = StreamInfo(
        name="hpi",
        stype="hpi",
        n_channels=12,
        sfreq=0,
        dtype=np.float32,
        source_id=source_id,
    )
    sinfo.set_channel_names(CH_NAMES["megin"])
    return sinfo


def test_stream_with_hpi(
    mock_megin_lsl_stream: DummyPlayer,
    hpi_stream_sinfo: StreamInfo,
    close_io: Callable[[], None],
) -> None:
    """Test an LSL stream with an attached HPI stream."""
    stream = StreamLSL(
        2, name=mock_megin_lsl_stream.name, source_id=mock_megin_lsl_stream.source_id
    ).connect(acquisition_delay=0.5)

    # create a mock HPI stream
    hpi_outlet = StreamOutlet(hpi_stream_sinfo)
    hpi_stream = StreamLSL(
        10, name=hpi_stream_sinfo.name, source_id=hpi_stream_sinfo.source_id
    ).connect(acquisition_delay=None)

    # check 'dev_head_t' before
    if check_version("mne", "1.11"):
        assert stream.info["dev_head_t"] is None
    else:
        assert stream.info["dev_head_t"] == Transform("meg", "head")  # identity

    # connect both and setup the callback
    stream.connect_hpi_stream(hpi_stream, format="megin")

    # check 'dev_head_t' after
    if check_version("mne", "1.11"):
        assert stream.info["dev_head_t"] is None
    else:
        assert stream.info["dev_head_t"] == Transform("meg", "head")  # identity

    # push a new HPI measurement
    hpi_outlet.push_sample(
        np.array([11, 12, 13, 21, 22, 23, 31, 32, 33, 1, 2, 3], dtype=np.float32)
    )
    time.sleep(0.1)  # wait a bit then acquire
    hpi_stream.acquire()

    # check 'dev_head_t' after the HPI measurement
    trans = np.array(
        [
            [11, 12, 13, 1],
            [21, 22, 23, 2],
            [31, 32, 33, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    assert stream.info["dev_head_t"] == Transform("meg", "head", trans)

    # push a new sample, with the first value changing (R11)
    hpi_outlet.push_sample(
        np.array([101, 12, 13, 21, 22, 23, 31, 32, 33, 1, 2, 3], dtype=np.float32)
    )
    time.sleep(0.1)  # wait a bit then acquire
    hpi_stream.acquire()

    # check 'dev_head_t' after the HPI measurement
    trans = np.array(
        [
            [101, 12, 13, 1],
            [21, 22, 23, 2],
            [31, 32, 33, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    assert stream.info["dev_head_t"] == Transform("meg", "head", trans)

    # clean-up
    stream.disconnect()
    hpi_stream.disconnect()
    close_io()


def test_check_hpi_ch_names() -> None:
    """Test the channel name validation."""
    with pytest.raises(RuntimeError, match="Expected HPI channel names"):
        check_hpi_ch_names(["foo"], "megin")
