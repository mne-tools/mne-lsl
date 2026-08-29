from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mne_lsl.viewer.controller import ChannelModel, ChannelsPage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qtpy.QtCore import QModelIndex

    from mne_lsl.stream import StreamLSL

# Channels of the mixed stream, one per code path of the model: four Volt channels
# declared in µV, one declared in mV so a reset has a multiplier to restore, one stim
# channel -- whose kind is Volts, an MNE quirk which makes its multiplier writable --
# and one misc channel, whose kind is not a physical quantity at all.
_NAMES = ["Fp1", "Fp2", "Cz", "Pz", "EOG", "ECG", "STI", "MISC"]
_TYPES = ["eeg", "eeg", "eeg", "eeg", "eog", "ecg", "stim", "misc"]
_UNITS = ["uv", "uv", "uv", "uv", "uv", "mv", "none", "none"]


class Emissions:
    """Record of everything a channel model emitted.

    The counts, rather than the payloads, are what the tests assert: the model's
    contract is that a bulk edit emits exactly one spanning ``dataChanged`` and exactly
    one coarse signal, and a per-row storm is the regression this catches.

    Parameters
    ----------
    model : ChannelModel
        The model to watch.

    Attributes
    ----------
    data : list of tuple of int
        The ``(first, last)`` row span of every ``dataChanged``.
    layout : int
        Number of ``layout_changed`` emissions.
    metadata : int
        Number of ``metadata_changed`` emissions.
    layout_qt : int
        Number of Qt ``layoutChanged`` emissions.
    reset : int
        Number of Qt ``modelReset`` emissions.
    """

    def __init__(self, model: ChannelModel) -> None:
        self.data: list[tuple[int, int]] = []
        self.layout = 0
        self.metadata = 0
        self.layout_qt = 0
        self.reset = 0
        model.dataChanged.connect(self._on_data)
        model.layout_changed.connect(self._on_layout)
        model.metadata_changed.connect(self._on_metadata)
        model.layoutChanged.connect(self._on_layout_qt)
        model.modelReset.connect(self._on_reset)

    def _on_data(self, first: QModelIndex, last: QModelIndex, *_args) -> None:
        self.data.append((first.row(), last.row()))

    def _on_layout(self) -> None:
        self.layout += 1

    def _on_metadata(self) -> None:
        self.metadata += 1

    def _on_layout_qt(self, *_args) -> None:
        self.layout_qt += 1

    def _on_reset(self) -> None:
        self.reset += 1


@pytest.fixture
def mixed_stream(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> StreamLSL:
    """Return an 8-channel stream covering every channel type the page can set."""
    stream, _ = lsl_stream(
        n_channels=len(_NAMES),
        ch_names=_NAMES,
        ch_types=_TYPES,
        ch_units=_UNITS,
    )
    return stream


@pytest.fixture
def model(
    mixed_stream: StreamLSL, flush_deletes: Callable[..., None]
) -> Generator[ChannelModel]:
    """Yield a channel model over the mixed stream."""
    built = ChannelModel(mixed_stream)
    yield built
    flush_deletes(built)


@pytest.fixture
def emissions() -> Callable[[ChannelModel], Emissions]:
    """Return a factory recording everything a model emits from now on."""
    return Emissions


@pytest.fixture
def make_page(
    flush_deletes: Callable[..., None],
) -> Generator[Callable[[ChannelModel], ChannelsPage]]:
    """Yield a factory building Channels pages, closed at teardown.

    Same shape as the trace display's factory: a page built in a test body needs the
    same close-and-delete teardown whether or not the test body reached its end.
    """
    created: list[ChannelsPage] = []

    def _make(model: ChannelModel, width: int = 430, height: int = 780) -> ChannelsPage:
        """Build one page over ``model`` and register it for teardown."""
        widget = ChannelsPage(model)
        widget.resize(width, height)
        created.append(widget)
        return widget

    yield _make
    for widget in reversed(created):
        widget.close()
    flush_deletes(*reversed(created))
    created.clear()


@pytest.fixture
def page(
    make_page: Callable[[ChannelModel], ChannelsPage], model: ChannelModel
) -> ChannelsPage:
    """Return a Channels page over the mixed-stream model, closed afterwards."""
    return make_page(model)


@pytest.fixture
def rows_of() -> Callable[[ChannelModel], list[str]]:
    """Return a helper listing a model's channel names in presentation order."""

    def _names(model: ChannelModel) -> list[str]:
        """Return the channel names of ``model``, in display order."""
        return [model.channel(row).name for row in range(model.rowCount())]

    return _names
