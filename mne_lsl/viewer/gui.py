from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from pyqtgraph import GraphicsLayoutWidget, PlotCurveItem, PlotItem
from qtpy.QtWidgets import QMainWindow

from .axis import ChannelAxis, TimeAxis

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
    def __init__(self, duration: float, ch_names: List[str]) -> None:
        self.axis_time = TimeAxis(duration)
        self.axis_channel = ChannelAxis(ch_names)
        super().__init__(axisItem={"bottom": self.axis_time, "left": self.axis_channel})
        self.getViewBox().invertY(True)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.getViewBox().disableAutoRange()
        self.axis_time.linkToView(self.getViewBox())
        self.axis_channel.linkToView(self.getViewBox())
        self.hideButtons()


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
