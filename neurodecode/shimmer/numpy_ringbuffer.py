#!/usr/bin/env python

# 20141021: rchava 
# from http://stackoverflow.com/questions/4151320/efficient-circular-buffer

import numpy as np


class RingBuffer(object):
    def __init__(self, size_max, default_value=0.0, dtype=float):
        """initialization"""
        self.size_max = size_max

        self._data = np.empty(size_max, dtype=dtype)
        self._data.fill(default_value)

        self.size = 0

    def append(self, value):
        """append an element"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value

        self.size += 1

        if self.size == self.size_max:
            self.__class__ = RingBufferFull

    def get_all(self):
        """return a list of elements from the oldest to the newest"""
        return (self._data)

    def get_partial(self):
        return (self.get_all()[0:self.size])

    def __getitem__(self, key):
        """get element"""
        return (self._data[key])

    def __repr__(self):
        """return string representation"""
        s = self._data.__repr__()
        s = s + '\t' + str(self.size)
        s = s + '\t' + self.get_all()[::-1].__repr__()
        s = s + '\t' + self.get_partial()[::-1].__repr__()
        return (s)


class RingBufferFull(RingBuffer):
    def append(self, value):
        """append an element when buffer is full"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value
