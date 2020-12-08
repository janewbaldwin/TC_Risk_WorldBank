#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import datetime
import module_riskModel as mrisk
import pickle
from netCDF4 import Dataset
from chaz import CLE15, utility

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
lat_poi = 18.975
lon_poi = 72.8258
radius = 150.  # km
er = 6371.0  # km

landmaskfile = '/home/clee/muri/windParametric/landmask.nc'
llon, llat, lldmask = mrisk.get_landmask(landmaskfile)
ldmask = lldmask[-12::-24, ::24][::-1, :]
xlong, xlat = np.meshgrid(llon[::24], llat[-12::-24])


# read data
fileName = '/data2/clee/bttracks/Allstorms.ibtracs_all.v03r10.nc'
nc = Dataset(fileName, 'r', format='NETCDF3_CLASSIC')
lon = np.hstack([nc.variables['source_lon'][:, :, 6].T,
                 nc.variables['source_lon'][:, :, 13].T])
lat = np.hstack([nc.variables['source_lat'][:, :, 6].T,
                 nc.variables['source_lat'][:, :, 13].T])
days = np.hstack([nc.variables['source_time'][:, :].T,
                  nc.variables['source_time'][:, :].T])  # 1858-11-17 00:00:00
wspd = np.hstack([nc.variables['source_wind'][:, :, 6].T,
                  nc.variables['source_wind'][:, :, 13].T])
pres = np.hstack([nc.variables['source_pres'][:, :, 6].T,
                  nc.variables['source_pres'][:, :, 13].T])
iIO = np.argwhere(np.array(lon[0, :]) != -30000.)[:, 0]
iIO = np.hstack([iIO[0:1295], iIO[1785:]])
lon_obs = np.array(lon[:, iIO])
lat_obs = np.array(lat[:, iIO])
wspd_obs = np.array(wspd[:, iIO])
lon_obs[lon_obs == -30000.] = np.float('nan')
lat_obs[lat_obs == -30000.] = np.float('nan')
wspd_obs[wspd_obs == -9990.] = np.float('nan')
ldmask_obs = np.zeros(lon_obs.shape)*np.float('nan')
for ii in range(lon_obs.shape[0]):
    for jj in range(lon_obs.shape[1]):
        if lon_obs[ii, jj] == lon_obs[ii, jj]:
            distance = np.sqrt(
                (lon_obs[ii, jj]-xlong)**2+(lat_obs[ii, jj]-xlat)**2)
            iy, ix = np.unravel_index(np.argmin(distance), distance.shape)
            iy1, ix1 = np.max([iy-2, 0]), np.max([ix-2, 0])
            iy2, ix2 = np.min([iy+2, distance.shape[0]]
                              ), np.min([ix+2, distance.shape[1]])
            ldmask_obs[ii, jj] = np.rint(
                np.nanmean(ldmask[iy1:iy2+1, ix1:ix2+1]))

a = np.unique(np.argwhere((lon_obs <= 80) & (lat_obs >= 0)
                          & (wspd_obs > 0) & (ldmask_obs < 2))[:, 1])
a = a[np.nanmax(wspd_obs[:, a], axis=0) >= 40]
a = np.unique(np.argwhere((lon_obs <= 80) & (lat_obs >= 0)
                          & (wspd_obs > 0) & (ldmask_obs < 2))[:, 1])
a = a[np.nanmax(wspd_obs[:, a], axis=0) >= 40]
lon_obs = lon_obs[:, a]
lat_obs = lat_obs[:, a]
wspd_obs = wspd_obs[:, a]
ipoi = utility.find_poi_Tracks(lon_obs, lat_obs, wspd_obs,
                               lon_poi, lat_poi, radius)


iIO = np.argwhere(np.array((lon[0, :]) != -30000.)
                  & (np.nanmax(wspd, axis=0) >= 35.))[:, 0]
lon = lon[:, iIO]
lat = lat[:, iIO]
wspd = wspd[:, iIO]
days = days[:, iIO]
iipoi = utility.find_poi_Tracks(lon, lat, wspd,
                                lon_poi, lat_poi, radius)
lon = lon[:, ipoi]
lat = lat[:, ipoi]
lon[lon == -3.00000000e+04] = np.float('nan')
lat[lat == -3.00000000e+04] = np.float('nan')
wspd[wspd == -9990.] = np.float('nan')
wspd = wspd[:, ipoi]
days = days[:, ipoi]
tt = np.empty(wspd.shape, dtype=object)
count = 0
for i in range(ipoi.shape[0]):
    for j in range(wspd.shape[0]):
        tt[j, count] = datetime.datetime(
            1858, 11, 17, 0, 0)+datetime.timedelta(days=days[j, i])
    count += 1
tr = np.zeros(wspd.shape)*np.float('nan')
trDir = np.zeros(wspd.shape)*np.float('nan')
for iS in range(ipoi.shape[0]):
    iT = np.argwhere(lon[:, iS] != -30000.).flatten()[-1]+1
    trDir[:iT, iS], tr[:iT, iS] =\
        utility.getStormTranslation(lon[:iT, iS], lat[:iT, iS], tt[:iT, iS])
rmax = utility.knaff15(wspd, lat)*1000.  # meter

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
            angle = np.arange(0., 360., 1.)/180.*np.pi+0.5*np.pi-trDir[ii, iS]
            rFactor = utility.translationspeedFactor(old_div(rr[:ir],rmax[ii, iS]))
            vt = np.array([tr[ii, iS]*np.cos(angle[id]) for id in range(360)])
            V = V+np.array([rFactor[iir]*vt for iir in range(ir)])
            londis = np.abs(old_div(2*np.pi*er*np.cos(lat[ii, iS]/180.*np.pi),360))
            theta, rr1 = np.meshgrid(np.arange(0., 360., 1.), rr[:ir]/1000.)
            dlon = old_div(rr1*np.cos(theta/180.*np.pi),111)
            dlat = old_div(rr1*np.sin(theta/180.*np.pi),londis)
            vlon = lon[ii, iS]+dlon
            vlat = lat[ii, iS]+dlat
            dummy = np.sqrt(((vlon-lon_poi)*londis)**2+((vlat-lat_poi)*111)**2)
            iir, iit = np.argwhere(dummy == dummy.min()).flatten()
            v_temp.append(V[iir, iit]*1.94384449)
            w_temp.append(np.nanmax(V)*1.94384449)
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
