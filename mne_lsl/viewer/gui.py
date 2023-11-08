from __future__ import annotations  # c.f. PEP 563, PEP 649

import numpy as np

from pyqtgraph import AxisItem, GraphicsLayoutWidget, PlotCurveItem, PlotItem
from qtpy.QtWidgets import QMainWindow


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
        self.setData(np.arange(data.shape[0]), data[:, idx])
        self.setPos(0, idx)


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = Viewer()
    app.exec()
