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
        self.show()


class ViewerGraphicsLayoutWidget(GraphicsLayoutWidget):
    """A GraphicsLayoutWidget with a PlotItem as its only child.

    In the future, additional PlotItems or ViewBoxes may be added as children to the
    LayoutWidget."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plot = ViewerPlotItem()
        self.addItem(self._plot)

        # add data traces
        data = np.random.randn(1000, 2) / 10  # stream buffer (n_samples, n_channels)
        self._plot.addItem(DataTrace(data=data, idx=0, parent=self._plot))
        self._plot.addItem(DataTrace(data=data, idx=1, parent=self._plot))


class ViewerPlotItem(PlotItem):
    def __init__(self, *args, **kwargs) -> None:
        self.axis_time = TimeAxis()
        self.axis_channel = ChannelAxis()
        super().__init__(
            axisItem={"bottom": self.axis_time, "left": self.axis_channel},
            *args,
            **kwargs,
        )
        self.getViewBox().invertY(True)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.getViewBox().disableAutoRange()
        self.hideButtons()


class TimeAxis(AxisItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(orientation="bottom", *args, **kwargs)


class ChannelAxis(AxisItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(orientation="left", *args, **kwargs)


class DataTrace(PlotCurveItem):
    def __init__(self, data: NDArray[float], idx: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._idx = idx
        self._data = data
        self.update_data()
        self.update_ypos()

    # TODO: We need to support channels with different scales, and this seems difficult
    # to do by "splitting" the y-range into multiple sub-axis with different scales.
    # Instead, a scaling factor should be applied to the data before plotting.
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
