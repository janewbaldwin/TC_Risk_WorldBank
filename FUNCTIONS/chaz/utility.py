#!/usr/bin/env python
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
from datetime import timedelta, datetime
from pygplib.util import argminDatetime
import matplotlib.pyplot as plt

er = 6371.0  # km


def knaff15(wspd, lat):
    '''
    wspd in kt, rmw in n. mile
    '''
    rmw = (218.3784-1.2014*wspd+(wspd/10.9844)**2
           - (wspd/35.3052)**3-145.5090*np.cos(old_div(lat, 180)*np.pi))
    rmw = rmw * 1.852  # (nautical mile to km)
    return rmw

def translationspeedFactor(r):
    """
    alpha = [0.4,0.7], r = [0,1], alpha = ar+b, a = 0.3, b = 0.4
    alpha =[0.7,0.5], r=[1,9] fix with exponetnal: exp(-0.314-0.042*np.array(r))
    """
    alpha = r*0.0
    alpha = np.exp(-0.314-0.042*r)
    alpha[r < 1] = 0.3*r[r < 1]+0.4
    #alpha[r>8] = 0.5
    return alpha


def getStormTranslation(lon, lat, time):

    timeInt = []
    lonInt = []
    latInt = []

    for iN in range(time.shape[0]-1):
        timeInt.append(time[iN])
        lonInt.append(lon[iN])
        latInt.append(lat[iN])
        delt = old_div(old_div((time[iN+1]-time[iN]).seconds, 60), 60)
        # print delt, lon[iN],lon[iN+1]
        if ((lon[iN+1] == lon[iN+1]) and (delt > 0)):
            inv = 1./np.float(delt)
            for iM in range(1, delt, 1):
                timeInt.append(time[iN]+timedelta(hours=iM))
                lonInt.append((1.-iM*inv)*lon[iN]+iM*inv*lon[iN+1])
                latInt.append((1.-iM*inv)*lat[iN]+iM*inv*lat[iN+1])
        else:
            timeInt.append(datetime(1800, 1, 1, 0, 0))
            lonInt.append(np.float('nan'))
            latInt.append(np.float('nan'))

    speed = np.zeros(lon.shape[0], dtype=float)+float('nan')
    sdir = np.zeros(lon.shape[0], dtype=float)+float('nan')
    count = 0
    for it in time:
        nup = argminDatetime(it+timedelta(hours=3), timeInt)
        ndn = argminDatetime(it-timedelta(hours=3), timeInt)
        londis = old_div(
            2*np.pi*er*np.cos(old_div(latInt[nup], 180)*np.pi), 360)
        dlon = lonInt[nup]-lonInt[ndn]
        dlat = latInt[nup]-latInt[ndn]
        dx = londis*(lonInt[nup]-lonInt[ndn])
        dy = 110*(latInt[nup]-latInt[ndn])
        distance = np.sqrt(dx*dx+dy*dy)  # km
        sdir[count] = np.arctan2(dlat, dlon)
        speed[count] = old_div(
            old_div(distance*1000./(nup-ndn), 60), 60)  # m/s
        count += 1

    return sdir, speed
