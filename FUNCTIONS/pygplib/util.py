#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from future.utils import raise_
from numpy import *
from datetime import datetime
from netCDF4 import Dataset
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bisect import bisect_left, bisect_right
import scipy
import copy

def argminDatetime(time0, time):
    """
    Written by Chia-Ying Lee (LDEO/Columbia).
    Returns the index of datetime array time for which
    the datetime time0 is nearest.
    """
    time = np.array(time)
    n0 = 0
    delt = np.abs(time[-1]-time[0])
    for n in range(time.size):
        if np.abs(time0-time[n]) < delt:
            delt = np.abs(time0-time[n])
            n0 = n
    return n0
