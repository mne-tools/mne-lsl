#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:11:49 2021

@author: mathieu
"""

import numpy as np

def fix_event_values(timearr):
    timearr[np.where(timearr==-1)] = 0
    return timearr
