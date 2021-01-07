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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from pyproj import Proj, transform
from pygplib3 import landfall as ld
from wind_reconstruct.w_profile_2 import W_profile # from Qidong
import math
import multiprocessing
from joblib import Parallel, delayed
import time
from os import path

# For timing the script
start_time = time.time()

# Bounding box for the Philippines
lonmin = 117.17427453
latmin = 5.58100332277
lonmax = 126.537423944
latmax = 18.5052273625

# Load subset data of landfalling storms over Philippines
dat = xr.open_dataset('ibtracs_landfall_philippines.nc')
lon = np.array(dat.lon)
lat = np.array(dat.lat)
wspd = np.array(dat.wspd)/1.944 # convert from kts to m/s
days = np.array(dat.days)
year = np.array(dat.year)
dat.close()

# Load land-sea mask
llon, llat, ldmask = ld.get_landmask('/home/clee/CHAZ/landmask.nc')
land = np.max(ldmask)
ocean = np.min(ldmask)

# Calculate track angle and track translation speed 
er = 6371.0  # earth's radius; km
lon_diff = lon[1:, :]-lon[0:-1, :]
lat_diff = lat[1:, :]-lat[0:-1, :]
londis = old_div(2*np.pi*er*np.cos(old_div(lat[1:, :],180)*np.pi),360) # longitude distance at a latitude; km
dx = londis*1000*lon_diff # meters
dy = 110.*1000*lat_diff # meters
time_diff = (days[1:, :] - days[0:-1, :])*24.*60*60 # seconds
tr = old_div(np.sqrt(dx**2+dy**2),(time_diff)) # track translation speed
trDir = np.arctan2(lat_diff, lon_diff) # track angle
# note: subtraction cuts off point at end, might need to add one final track direction and speed point if get errors later

# Interpolate to 15-min timesteps
nscale = 24 # convert from 6-hr to 15-min timesteps (factor of 6*4=24)
lon = ld.rescale_matrix(lon,nscale,0) # int for time interpolated 
lat = ld.rescale_matrix(lat,nscale,0)
wspd = ld.rescale_matrix(wspd,nscale,0)
days = ld.rescale_matrix(days,nscale,0)
tr = ld.rescale_matrix(tr,nscale,0)
trDir = ld.rescale_matrix(trDir,nscale,0)

# Retrieve times of landfall for the WNP
iSlandfall = ld.get_landfall_stormID(lon,lat,wspd,llon,llat,ldmask,land,np.min(ldmask))
landfall_times = ld.get_landfall_storm_time(iSlandfall,lon,lat,wspd,llon,llat,ldmask,land,ocean,24)
nSlandfall_first = landfall_times[0] # index of storms that make first landfall (why different than nSlandfall?)
iTlandfall_first = landfall_times[1] # time of making first landfall
nSlandfall_all = landfall_times[2] # index of storms that make any landfall (ie storm would repeat if makes multiple landfalls)
iTlandfall_all = landfall_times[3] # time of making that landfall

# Find times that storms make landfall in the Philippines
nSlandfall_all_phi = []
iTlandfall_all_phi = []
for j in range(np.shape(nSlandfall_all)[0]):
    nS = nSlandfall_all[j]
    iT = iTlandfall_all[j]
    lon_landfall = lon[iT,nS]
    lat_landfall = lat[iT,nS]
    if lonmin <= lon_landfall <= lonmax and latmin <= lat_landfall <= latmax:
        nSlandfall_all_phi.append(nSlandfall_all[j])
        iTlandfall_all_phi.append(iTlandfall_all[j])

# Remove duplicate storms (storms that made landfall in Philippines twice)
nSlandfall_phi = list(dict.fromkeys(nSlandfall_all_phi))

# Select time of first landfall for each of the Philippines storms
iTlandfall_first_phi = []
for i in range(np.shape(lon)[1]):
    j = np.where(np.array(nSlandfall_all_phi)==i)[0][0]
    iTlandfall_first_phi.append(iTlandfall_all_phi[j])
    
# For each storm select time points of landfall, including possibility for second landfall and potential overlap
# iTlandfall_forwindfield_phi

days_before_landfall = 1
days_post_landfall = 1
timesteps_before_landfall = days_before_landfall*4*24 # 4 15-min increments per hour, 24 hours per day
timesteps_post_landfall = days_post_landfall*4*24 # 4 15-min increments per hour, 24 hours per day

iTlandfall_forwindfield_phi = []
nSlandfall_forwindfield_phi = []
for i in range(np.shape(lon)[1]):
    nSlandfall_forwindfield_phi.append(i)
    indices_landfalls = np.where(np.array(nSlandfall_all_phi)==i)[0] # different landfalls per storm
    nlandfalls = np.shape(indices_landfalls)[0]
    if nlandfalls == 1:
        iTlandfall = iTlandfall_all_phi[indices_landfalls[0]]
        iT = np.arange(iTlandfall-timesteps_before_landfall,iTlandfall+timesteps_post_landfall+1,1)
        iTlandfall_forwindfield_phi.append(list(iT))
    if nlandfalls > 1:
        iTs = np.array([],dtype=int)
        for ii in indices_landfalls:
            iTlandfall = iTlandfall_all_phi[ii]
            iT = np.arange(iTlandfall-timesteps_before_landfall,iTlandfall+timesteps_post_landfall+1,1)
            iTs = np.concatenate((iTs,iT),axis=0)
        iTs = np.unique(iTs) # remove wind field points that repeat
        iTlandfall_forwindfield_phi.append(list(iTs))
    iTlandfall_forwindfield_phi[i] = list(filter(lambda x : x > 0, iTlandfall_forwindfield_phi[i])) # remove negative numbers from list https://www.geeksforgeeks.org/python-remove-negative-elements-in-list/ 
            
# Aggregate only storm points (lat,lon,wspd) where might be making landfall
wspd_landfall = []
lon_landfall = []
lat_landfall = []
days_landfall = []
tr_landfall = []
trDir_landfall = []
year_landfall = []

for nS in range(np.shape(lon)[1]):
    iT = iTlandfall_forwindfield_phi[nS]
    wspd_landfall.append(wspd[iT,nS])
    lon_landfall.append(lon[iT,nS])
    lat_landfall.append(lat[iT,nS])
    days_landfall.append(days[iT,nS])
    tr_landfall.append(tr[iT,nS])
    trDir_landfall.append(trDir[iT,nS])
    
# Distance function
# Center of storm = lat, lon
# Calculate distance between center point and each grid point
# adopted from: https://kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python
# radius of earth in km so outputs distance in km

def distancefrompoint(lon, lat, X1, Y1):
    R = 6371.0 #radius of the Earth km

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = np.radians(Y1)
    lon2 = np.radians(X1)

    # change in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Set up function to run wind fields in parallel
# 32 processors total available


# Calculate wind field with asymmetry, subtracting max rFactor*vt from wspdi before calculating profile:
def windfield(lon_nS,lat_nS,wspd_nS,rmax_nS,tr_nS,trDir_nS,i):
    loni = lon_nS[i]
    lati = lat_nS[i]
    wspdi = wspd_nS[i]
    rmaxi = rmax_nS[i]
    tri = tr_nS[i]
    trDiri = trDir_nS[i]
    
    # Calculate tangential wind
    angle = np.arctan2((Y1-lati),(X1-loni)) - trDiri # define angle relative to track direction 
    vt = -tri*np.cos(np.pi/2 - angle) # calculate tangential wind; remove minus if southern hemisphere
    
    # Calculate distance from center of storm
    distance = distancefrompoint(loni, lati, X1, Y1) # distance in km
    
    # Round distance values to nearest whole number
    distance = distance.astype(int)
    
    # Calculate rFactor to modulate track correction
    rFactor = utility.translationspeedFactor(old_div(distance,rmaxi))
    asymcorrec = rFactor*vt
    max_asymcorrec = 0.7*tri # 0.7 from utility.translationspeedFactor structure. tri = max vt. Alternatively could do np.max(rFactor)*np.max(vt), but this should be faster and more exact.
    
    # Calculate Willoughby Profile
    radius_max = 500
    radius_precision = 1
    profile = W_profile(lati, rmaxi, wspdi-max_asymcorrec, radius_max, radius_precision)
    radius = np.arange(0,radius_max + radius_precision, radius_precision)
    
    # Create dict look-up table from Willoughby Profile
    wspdlookup = dict(zip(radius, profile))

    # Remap radii to windspeed
    wspdmap = np.zeros(np.shape(distance))
    for r in radius:
        wspdmap[np.where(distance == r)] = wspdlookup[r]
    
    #Add track direction correction
    wspdmap = wspdmap + asymcorrec
    
    # Set to 0 outside radius_max
    wspdmap[np.where(distance > radius_max)] = 0 # added 10-27-20
    
    return wspdmap


# Define x-y grid to put profile on
X = np.arange(116.5,127.6,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
Y = np.arange(4.5,19.1,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
X1, Y1 = np.meshgrid(X,Y)

# Determine the max iT_lengths
iT_lengths = []
for i in np.arange(0,len(iTlandfall_forwindfield_phi),1):
    iT_lengths.append(len(iTlandfall_forwindfield_phi[i]))
iT_length_max = np.max(iT_lengths)


missed_tries = 0
for nS in np.arange(0,len(wspd_landfall)+1,1):
    print(nS)
    
    # Select data for 1 storm
    notnans = ~np.isnan(tr_landfall[nS][:]) # remove nan points from TC dissipating sooner than 1 day after first landfall, and for last time step that track speed can't be calculated for
    wspd_nS = wspd_landfall[nS][notnans]
    lon_nS = lon_landfall[nS][notnans]
    lat_nS = lat_landfall[nS][notnans]
    days_nS = days_landfall[nS][notnans]
    tr_nS = tr_landfall[nS][notnans]
    trDir_nS = trDir_landfall[nS][notnans]
    
    # Calculate radius of maximum wind
    rmax_nS = utility.knaff15(wspd_nS*1.944, lat_nS)  #  wspd should be input in kts, outputs in km
    rmax_min = 20 # minimum rmax used if Knaff15 produces values that are unrealistically small; km
    rmax_nS[np.where(rmax_nS<rmax_min)] = rmax_min # 6-12-20: set places with radii less than 0 to 20 km to fix convergence
    
    # Calculate wind fields in parallel
    stormpoints = np.shape(wspd_nS)[0]  
    try:
        wspdmaps = Parallel(n_jobs=16, prefer="threads")(delayed(windfield)(lon_nS,lat_nS,wspd_nS,rmax_nS,tr_nS,trDir_nS,i) for i in range(stormpoints))    
        wspdmaps = np.expand_dims(wspdmaps,axis = 0)
        swath = np.nanmax(wspdmaps, axis = 1) # Calculate swath over windfields; nanmax to ignore timepoints that don't have windfields
        
        # Days
        days_fornc = days_landfall[nS][notnans]
        
         # Create swath dataset
        ds = xr.Dataset(
         {"swath": (("nS", "lat", "lon"), swath),
          "days": (("nS","iT"), np.expand_dims(days_fornc,axis=0)),
          "year": (("nS"), [year[nS]])},
             coords={
            "nS":np.array([nS]),
            "iT":np.arange(stormpoints),
            "lat": Y,
            "lon": X,
         },
         )
    
        #Write to netcdf
        direc = '/data2/jbaldwin/WINDFIELDS/PHI_SWATHS/'
        filename = 'wspd_phi_swaths.nc'
        #ds.to_netcdf(direc+filename,mode='a',unlimited_dims = ["nS"])
        if path.exists(direc+filename): # concatenate if file exists
             ds_swaths = xr.open_dataset(direc+filename)
             ds_swaths2 = xr.concat([ds_swaths,ds],dim='nS')
             ds_swaths.close() # need to do this before can save out to same file
             ds_swaths2.to_netcdf(direc+filename,mode='w')
        else: #if not create new file
             ds.to_netcdf(direc+filename,mode='w',unlimited_dims = ["nS"])
            
    except:
        missed_tries += 1
    
print("My program took", time.time() - start_time, "to run and had ", missed_tries, " missed tries.")
