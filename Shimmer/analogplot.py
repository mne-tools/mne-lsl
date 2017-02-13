################################################################################
# class that holds analog data for N samples. Extracted from:
# showdata.py
#
# Display analog data from Arduino using Python (matplotlib)
# 
# electronut.in
#
# http://www.instructables.com/id/Plotting-real-time-data-from-Arduino-using-Python-/
################################################################################
from matplotlib import pyplot as plt
from collections import deque


class AnalogData:
    # constr
    def __init__(self, maxLen):
        self.ax = deque([0.0] * maxLen)
        self.ay = deque([0.0] * maxLen)
        self.maxLen = maxLen

    # ring buffer
    def addToBuf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert (len(data) == 2)
        self.addToBuf(self.ax, data[0])
        self.addToBuf(self.ay, data[1])


# plot class
class AnalogPlot:
    # constr
    def __init__(self, analogData):
        # set plot to animated
        plt.ion()
        self.axline, = plt.plot(analogData.ax)
        self.ayline, = plt.plot(analogData.ay)
        plt.ylim([0, 3000])

    # update plot
    def update(self, analogData):
        self.axline.set_ydata(analogData.ax)
        self.ayline.set_ydata(analogData.ay)
        plt.draw()
