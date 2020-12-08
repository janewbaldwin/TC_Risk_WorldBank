#!/usr/bin/env python
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import datetime
import pickle
from netCDF4 import Dataset
import sys
import matplotlib.pyplot as plt
from chaz import CLE15, utility
from pygplib import readbst
import xarray as xr
import pandas as pd

### constant from Dan Chavas ###
fcor = 5.e-5  # [s-1] {5e-5}; Coriolis parameter at storm center
# Environmental parameters
# Outer region
# [-] {1}; 0 : Outer region Cd = constant (defined on next line); 1 : Outer region Cd = f(V) (empirical Donelan et al. 2004)
Cdvary = 1
# [-] {1.5e-3}; ignored if Cdvary = 1; surface momentum exchange (i.e. drag) coefficient
Cd = 1.5e-3
# [ms-1] {2/1000; Chavas et al 2015}; radiative-subsidence rate in the rain-free tropics above the boundary layer top
w_cool = 2./1000

# Inner region
# [-] {1}; 0 : Inner region Ck/Cd = constant (defined on next line); 1 : Inner region Ck/Cd = f(Vmax) (empirical Chavas et al. 2015)
CkCdvary = 1
# [-] {1}; ignored if CkCdvary = 1; ratio of surface exchange coefficients of enthalpy and momentum; capped at 1.9 (things get weird >=2)
CkCd = 1.

# Eye adjustment
eye_adj = 0  # [-] {1}; 0 = use ER11 profile in eye; 1 = empirical adjustment
# [-] {.15; empirical Chavas et al 2015}; V/Vm in eye is reduced by factor (r/rm)^alpha_eye; ignored if eye_adj=0
alpha_eye = .15
###

# for Mumbai
cityName = 'Bermuda'
lat_poi = 32.307
lon_poi = -64.7505+360.
radius = 300.  # km
er = 6371.0  # km

# read data
fileName = '/data2/clee/bttracks/Allstorms.ibtracs_all.v03r10.nc'
ibtracs = readbst.read_ibtracs(fileName, 'atl')
ipoi = utility.find_poi_Tracks(ibtracs.lon[:, :], ibtracs.lat[:, :], ibtracs.wspd[:, :],
                               lon_poi, lat_poi, radius)
lon = ibtracs.lon[:, ipoi]
lat = ibtracs.lat[:, ipoi]
wspd = ibtracs.wspd[:, ipoi]
days = ibtracs.days[:, ipoi]
dist2land = ibtracs.dist2land[:, ipoi]
year = ibtracs.year[ipoi]
tt = np.empty(wspd.shape, dtype=object)
count = 0
for i in range(ipoi.shape[0]):
    for j in range(wspd.shape[0]):
        if days[j, i] == days[j, i]:
            tt[j, count] = datetime.datetime(
                1858, 11, 17, 0, 0)+datetime.timedelta(days=days[j, i])
    count += 1
lon_diff = lon[1:, :]-lon[0:-1, :]
lat_diff = lat[1:, :]-lat[0:-1, :]
londis = old_div(2*np.pi*er*np.cos(old_div(lat[1:, :],180)*np.pi),360)
dx = londis*lon_diff
dy = 110.*lat_diff
days_diff = (days[1:, :] - days[0:-1, :])*24.
tr = old_div(np.sqrt(dx**2+dy**2),(days_diff)) # track translation speed?
trDir = np.arctan2(lat_diff, lon_diff)
#tr1 = np.zeros(wspd.shape)*np.float('nan')
#trDir1 = np.zeros(wspd.shape)*np.float('nan')
# for iS in range(ipoi.shape[0]):
#    iT = np.argwhere(np.isnan(lon[:,iS])).flatten()[-1]+1
#    trDir1[:iT,iS],tr1[:iT,iS] =\
#        utility.getStormTranslation(lon[:iT,iS],lat[:iT,iS],tt[:iT,iS])
rmax = utility.knaff15(wspd, lat)*1000.  # meter
# tr1 = tr1*3.6 #km/hr

v_poi = []
wspd_poi = []
rmw_poi = []
wspd_poi_v2 = []
lon1 = []
lat1 = []
for iS in range(lon.shape[1]):
    iipoi = utility.find_timing_Tracks(
        lon[:, iS], lat[:, iS], wspd[:, iS], lon_poi, lat_poi, radius)
    if iipoi.size > 0:
        plt.plot(lon[:, iS], lat[:, iS])
        print(tt[0, iS], wspd[iipoi, iS])
        londis = old_div(2*np.pi*er*np.cos(old_div(lat[iipoi, iS],180)*np.pi),360)
        dx = londis*(lon[iipoi, iS]-lon_poi)
        dy = 110*(lat[iipoi, iS]-lat_poi)
        distance = np.sqrt(dx*dx+dy*dy)
        distance[distance != distance] = radius+10000
        # for ii in iipoi:
        #	  if wspd[ii,iS]==-9990.:
        #	     wspd[ii,iS] = 25.
        wspd_poi.append(np.nanmax(wspd[iipoi, iS]))
        lon1.append(lon[iipoi[np.argmax(wspd[iipoi, iS])], iS])
        lat1.append(lat[iipoi[np.argmax(wspd[iipoi, iS])], iS])
        rmw_poi.append(rmax[iipoi[np.argmax(wspd[iipoi, iS])], iS])
        print(iS, iipoi)
        v_temp = []
        w_temp = []
        for ii in iipoi:
            if not np.isnan(tr[ii, iS]):
                wspd_az = wspd[ii, iS]/1.94384449 - tr[ii, iS]*0.7  # in ms-1
            else:
                wspd_az = wspd[ii, iS]/1.94384449

            # need to use ms-1
            rr, VV, r0, rmerge, Vmerge = CLE15.ER11E04_nondim_rmaxinput(wspd_az,
                                                                        rmax[ii,
                                                                             iS], fcor, Cdvary,
                                                                        Cd, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye)
            # stop at 300Km
            ir = np.argwhere(rr <= (radius+50)*1000.)[-1, 0]
            V = np.zeros([ir, 360])
            V = np.reshape(np.repeat(VV[:ir], 360), V.shape)
            angle = np.arange(0., 360., 1.)/180.*np.pi+0.5*np.pi-trDir[ii, iS] # define angle relative to track direction 
            rFactor = utility.translationspeedFactor(old_div(rr[:ir],rmax[ii, iS]))
            vt = np.array([tr[ii, iS]*np.cos(angle[id]) for id in range(360)])
            V = V+np.array([rFactor[iir]*vt for iir in range(ir)])
            londis = np.abs(old_div(2*np.pi*er*np.cos(lat[ii, iS]/180.*np.pi),360)) # distance of 1 deg longitude at storm latitude
            theta, rr1 = np.meshgrid(np.arange(0., 360., 1.), rr[:ir]/1000.) # make mesh grid (360 deg x radii) of values of angle and radius in wind field
            dlon = old_div(rr1*np.cos(theta/180.*np.pi),111) # translate distance along radius in direction to deg longitude
            dlat = old_div(rr1*np.sin(theta/180.*np.pi),londis) # translate distance along radius in direction to deg latitude
            vlon = lon[ii, iS]+dlon # longitude of each wind field point
            vlat = lat[ii, iS]+dlat # latitude of each wind field point
            dummy = np.sqrt(((vlon-lon_poi)*londis)**2+((vlat-lat_poi)*111)**2) # distance between wind field grid points and POI
            iir, iit = np.argwhere(dummy == dummy.min()).flatten() # find place in wind field closest to POI
            v_temp.append(V[iir, iit]*1.94384449) # 1.94384 --> convert m/s to knots
            w_temp.append(np.nanmax(V)*1.94384449) # 1.94384 --> convert m/s to knots
            # v_poi = griddata((vlon.flatten(),vlat.flatten()),V.flatten(),(lon_poi,lat_poi)
        v_poi.append(np.nanmax(np.array(v_temp)))
        wspd_poi_v2.append(
            np.nanmax([np.nanmax(np.array(w_temp)), np.nanmax(wspd[iipoi, iS])]))
    else:
        v_poi.append(float('nan'))
        wspd_poi.append(float('nan'))
        rmw_poi.append(float('nan'))
        wspd_poi_v2.append(float('nan'))
        lon1.append(float('nan'))
        lat1.append(float('nan'))

v_poi = np.array(v_poi)
wspd_poi = np.array(wspd_poi)
with open('Obs_Mumbai_poi_knaff15_onePt.pik', 'w+') as f:
    pickle.dump(lon, f)
    pickle.dump(lat, f)
    pickle.dump(wspd, f)
    pickle.dump(v_poi, f)
    pickle.dump(wspd_poi, f)
    pickle.dump(wspd_poi_v2, f)
    pickle.dump(rmw_poi, f)
f.close()
