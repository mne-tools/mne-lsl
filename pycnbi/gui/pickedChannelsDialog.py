from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QPushButton, QLineEdit, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from pycnbi import pycnbi_config

########################################################################
class PickChannelsDialog(QDialog):
    """
    Defines the Dialog for selected channels for different cases (picked, excluded, filter...)
    """
    
    signal_paramChanged = pyqtSignal(list)

    # ----------------------------------------------------------------------
    def __init__(self, channels, selected=[], title="Pick channels"):
        """
        Constructor.
        """
        super().__init__()

        self.setWindowTitle(title)
        self.initial_selection = selected
        vbox = QVBoxLayout(self)
        self.channels = QListWidget()
        self.channels.insertItems(0, channels)
        self.channels.setSelectionMode(QListWidget.ExtendedSelection)

        for i in range(self.channels.count()):
            if self.channels.item(i).data(0) in selected:
                self.channels.item(i).setSelected(True)

        vbox.addWidget(self.channels)
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Ok |
                                          QDialogButtonBox.Cancel)
        vbox.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        self.buttonbox.accepted.connect(self.on_modify)

        self.channels.itemSelectionChanged.connect(self.toggle_buttons)
        self.toggle_buttons()  # initialize OK button state


    @pyqtSlot()
    # ----------------------------------------------------------------------
    def toggle_buttons(self):
        """
        Toggle OK button.
        """
        selected = [item.data(0) for item in self.channels.selectedItems()]

        if selected != self.initial_selection:
            self.buttonbox.button(QDialogButtonBox.Ok).setEnabled(True)
            self.selected = selected
        else:
            self.buttonbox.button(QDialogButtonBox.Ok).setEnabled(False)

    # ----------------------------------------------------------------------
    def on_modify(self):
        """"""
        self.signal_paramChanged.emit(self.selected)


#######################################################################
class Channel_Select(QObject):
    """
    Contains a pushButton, a lineEdit and a PickChannelsDialog instance.
    The pushButton opens the PickChannelsDialog. The user can select the
    channels of interest and they will be displayed in the lineEdit.
    """

    signal_paramChanged = pyqtSignal([str, list])

    #----------------------------------------------------------------------
    def __init__(self, key, channels, selected):
        """
        Constructor
        
        channels = full channels list, specific to the eeg headset
        selected = channels pre-selection
        """
        super().__init__()

        self.layout = QHBoxLayout()
        self.key = key

        pushButton = QPushButton('Select')
        pushButton.clicked.connect(self.on_pushButton)
        self.layout.addWidget(pushButton)

        self.lineEdit = QLineEdit(str(selected))
        self.lineEdit.setReadOnly(True)
        self.layout.addWidget(self.lineEdit)

        self.pickChannelsDialog = PickChannelsDialog(channels, selected)
        self.pickChannelsDialog.hide()
        self.pickChannelsDialog.signal_paramChanged.connect(self.on_modify)


    # @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_pushButton(self):
        """
        Shows the PickChannelsDialog when the pushButton is clicked
        """
        self.pickChannelsDialog.show()

    # ----------------------------------------------------------------------
    # @pyqtSlot(list)
    def on_modify(self, new_value):
        """
        Modify the lineEdit according to the newly selected channels
        """
        self.lineEdit.setText(str(new_value))
        self.signal_paramChanged.emit(self.key, new_value)