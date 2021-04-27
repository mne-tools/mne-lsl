'''
Preprocessing.

It supports temporal, spatial, notch filtering, downsampling, rereferencing.
This module contains the find_event_channel using heuristic methods.
'''

from .preprocess import preprocess, find_event_channel, rereference