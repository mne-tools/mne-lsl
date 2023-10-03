from abc import ABC, abstractmethod

from qtpy.QtWidgets import QMainWindow


class _metaclass_ControlGUI(type(QMainWindow), type(ABC)):
    pass


class _ControlGUI(QMainWindow, ABC, metaclass=_metaclass_ControlGUI):
    """Class representing a base controller GUI.

    Parameters
    ----------
    scope : Scope
        Scope connected to a StreamInlet acquiring the data and applying
        filtering. The scope has a buffer of _BUFFER_DURATION seconds
        (default: 30s).
    """

    @abstractmethod
    def __init__(self, scope):
        super().__init__()
        self._scope = scope

    @abstractmethod
    def _load_gui(self):
        """Load the GUI."""
        pass

    @abstractmethod
    def _init_backend(self):
        """Initialize the backend."""
        pass

    def closeEvent(self, event):
        """Event called when closing the GUI."""
        event.accept()

    @abstractmethod
    def _connect_signals_to_slots(self):
        """Event handler. Connect QT signals to slots."""
        pass

    # --------------------------------------------------------------------
    @property
    def scope(self):
        """Measuring scope."""
        return self._scope

    @property
    @abstractmethod
    def backend(self):
        """Display backend."""
        pass

    @property
    @abstractmethod
    def ui(self):
        """Control UI."""
        pass
