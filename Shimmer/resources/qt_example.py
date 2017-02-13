#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, math
from PyQt4 import QtGui, QtCore
from time import *


class RASER(QtGui.QWidget):
    def __init__(self):
        super(RASER, self).__init__()
        self.initUI()

    def initUI(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_loop)
        timer.start(10);  # call self.update_loop every 10 ms

        self.tt = time()

    def update_loop(self):
        # Do iterative stuff here
        self.tt2 = time()
        print self.tt2 - self.tt
        self.tt = self.tt2


def main():
    app = QtGui.QApplication(sys.argv)
    ex = RASER()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
