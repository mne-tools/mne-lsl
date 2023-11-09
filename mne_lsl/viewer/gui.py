from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np

from pyqtgraph import AxisItem, GraphicsLayoutWidget, PlotCurveItem, PlotItem
from qtpy.QtWidgets import QMainWindow

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Viewer(QMainWindow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setCentralWidget(ViewerGraphicsLayoutWidget())

        # mock data
        ch_names = ["Cz", "Fz"]
        ch_types = ["eeg", "eeg"]
        ch_units = ["microvolts", "microvolts"]
        data = np.random.randn(1000, 2) / 10  # stream buffer (n_samples, n_channels)

        self.centralWidget().plot.addItem(
            DataTrace(data=data, idx=0, parent=self.centralWidget().plot)
        )
        self.centralWidget().plot.addItem(
            DataTrace(data=data, idx=1, parent=self.centralWidget().plot)
        )

        self.show()


class ViewerGraphicsLayoutWidget(GraphicsLayoutWidget):
    """A GraphicsLayoutWidget with a PlotItem as its only child.

    In the future, additional PlotItems or ViewBoxes may be added as children to the
    LayoutWidget."""

    def __init__(self) -> None:
        super().__init__()
        self._plot = ViewerPlotItem(duration=1)
        self.addItem(self._plot)

    @property
    def plot(self) -> ViewerPlotItem:
        """Main signal plotting object, with axis, aa ViewBox and data traces."""
        return self._plot


class ViewerPlotItem(PlotItem):
    def __init__(self, duration: float) -> None:
        self.axis_time = TimeAxis(duration)
        self.axis_channel = ChannelAxis()
        super().__init__(axisItem={"bottom": self.axis_time, "left": self.axis_channel})
        self.getViewBox().invertY(True)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.getViewBox().disableAutoRange()
        self.axis_time.linkToView(self.getViewBox())
        self.axis_channel.linkToView(self.getViewBox())
        self.hideButtons()


class TimeAxis(AxisItem):
    """Time axix, from 1 to 60 seconds."""

    def __init__(self, duration: float) -> None:
        self._duration = duration
        super().__init__(orientation="bottom")
        # tick spacing and formatting
        self._major = 1.0
        self._minor = 0.25
        self.enableAutoSIPrefix(False)
        self.setTickSpacing(major=self.major, minor=self.minor)
        self.update_ticks()

    def update_ticks(self) -> None:
        super().setTicks(
            [
                [(elt, str(elt)) for elt in np.arange(self.duration, self.major)],
                [(elt, str(elt)) for elt in np.arange(self.duration, self.minor)],
            ]
        )

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        assert 1 <= duration <= 60, "'duration' should be between 1 and 60 seconds."
        self._duration = duration
        self.update_ticks()

    @property
    def major(self) -> float:
        """Spacing between major ticks."""
        return self._major

    @major.setter
    def major(self, major: float) -> float:
        assert 0 < major, "'major' should be a positive number."
        self._major = major
        self.setTickSpacing(major=self.major, minor=self.minor)
        self.update_ticks()

    @property
    def minor(self) -> float:
        """Spacing between minor ticks."""
        return self._minor

    @minor.setter
    def minor(self, minor: float) -> float:
        assert 0 < minor, "'minor' should be a positive number."
        self._minor = minor
        self.setTickSpacing(major=self.major, minor=self.minor)
        self.update_ticks()


class ChannelAxis(AxisItem):
    def __init__(self) -> None:
        super().__init__(orientation="left")


class DataTrace(PlotCurveItem):
    def __init__(self, data: NDArray[float], idx: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._idx = idx
        self._data = data
        self.update_data()
        self.update_ypos()

    # TODO: We need to support channels with different scales, and this seems difficult
    # to do by "splitting" the y-range into multiple sub-axis with different scales.
    # Instead, a scaling factor should be applied to the data array before plotting.
    def update_ypos(self) -> None:
        self._ypos = self._idx
        self.setPos(0, self._ypos)

    def update_data(self) -> None:
        self.setData(np.arange(self._data.shape[0]), self._data[:, self._idx])


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = Viewer()
    app.exec()
