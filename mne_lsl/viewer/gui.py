from __future__ import annotations  # c.f. PEP 563, PEP 649

from pyqtgraph import AxisItem, GraphicsLayoutWidget, PlotCurveItem, PlotItem
from qtpy.QtWidgets import QMainWindow


class Viewer(QMainWindow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setCentralWidget(ViewerGraphicsLayoutWidget())


class ViewerGraphicsLayoutWidget(GraphicsLayoutWidget):
    """A GraphicsLayoutWidget with a PlotItem as its only child.

    In the future, additional PlotItems or ViewBoxes may be added as children to the
    layout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layout = self.addLayout()
        self._view_box = self._layout.addViewBox(invertY=True)
        self._plot = ViewerPlotItem(viewBox=self._view_box)


class ViewerPlotItem(PlotItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._axis_time = TimeAxis()
        self._axis_channel = ChannelAxis()
        self.setAxisItems({"bottom": self._axis_time, "left": self._axis_channel})
        self.hidebuttons()


class TimeAxis(AxisItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(orientation="bottom", *args, **kwargs)


class ChannelAxis(AxisItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(orientation="left", *args, **kwargs)


class DataTrace(PlotCurveItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
