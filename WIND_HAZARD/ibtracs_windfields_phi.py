#!/usr/bin/env python
# coding: utf-8

# # Run these three cells first:

# In[7]:


#reset


# In[8]:


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
from chaz import utility
from pygplib3 import readbst
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from pyproj import Proj, transform
from pygplib3 import landfall as ld
from wind_reconstruct.w_profile_2 import W_profile # from Qidong
from wind_reconstruct.w_profile import W_profile as W_profile_old # from Qidong
import math
import multiprocessing
from joblib import Parallel, delayed
import time
from os import path
from tcrisk.hazard import windfield, landfall_in_box, timepoints_around_landfall

# For timing the script
start_time = time.time()


# ## Step 1a: Define regional bounding box for Philippines

# In[5]:


# Bounding box for the Philippines
lonmin = 117.17427453
latmin = 5.58100332277
lonmax = 126.537423944
latmax = 18.5052273625


# ## Step 1b: Load data for TC tracks.

# In[6]:


# Load data from file
filename = '/data2/clee/bttracks/IBTrACS.ALL.v04r00.nc'
ibtracs = readbst.read_ibtracs_v4(filename,'wnp',2) # gap = 2 to convert 3-hourly to 6-hourly

# Extract variables
lon = ibtracs.lon[:]
lat = ibtracs.lat[:]
wspd = ibtracs.wspd[:] # wind speed in knots
days = ibtracs.days[:] # date in days
dist2land = ibtracs.dist2land[:]
year = ibtracs.year[:]


# ## Step 1c: Interpolate data from 6-hour to 15-min timesteps.

# In[11]:


# Interpolate to 15-min timesteps
nscale = 24 # convert from 6-hr to 15-min timesteps (factor of 6*4=24)
lon_int = ld.rescale_matrix(lon,nscale,0) # int for time interpolated 
lat_int = ld.rescale_matrix(lat,nscale,0)
wspd_int = ld.rescale_matrix(wspd,nscale,0)
days = np.where(days==-9999000., np.nan, days)
days_int = ld.rescale_matrix(days,nscale,0)


# ## Step 1d: Calculate which storms make landfall in the Philippines, and subset only those storms.
# Note: takes a couple minutes.

# In[14]:


# Calculate which storms and when make landfall in Philippines
llon_midpoint = 180
nSlandfall_all_phi, iTlandfall_all_phi, nSlandfall_phi = landfall_in_box(lonmin,lonmax,latmin,latmax,lon_int,lat_int,wspd_int,llon_midpoint)


# In[15]:


# Select data only for storms that make landfall in the Philippines
# (for normal timesteps and interpolated timestep data)
lon_phi = lon[:,nSlandfall_phi]
lat_phi = lat[:,nSlandfall_phi]
wspd_phi = wspd[:,nSlandfall_phi]
days_phi = days[:,nSlandfall_phi]
year_phi = year[nSlandfall_phi]

lon_int_phi = lon_int[:,nSlandfall_phi]
lat_int_phi = lat_int[:,nSlandfall_phi]
wspd_int_phi = wspd_int[:,nSlandfall_phi]
days_int_phi = days_int[:,nSlandfall_phi]


# ## Step 1e: Save out data for storms that make landfall in the Philippines.

# In[73]:


# Save out data of Philippines landfalling TCs
ds = xr.Dataset(
    {"lon": (("iT","nS"), lon_phi),
     "lat": (("iT","nS"), lat_phi),
     "wspd": (("iT","nS"), wspd_phi), # maximum sustained wind speed in m/s
     "days": (("iT","nS"), days_phi),
     "year": (("nS"), year_phi)},
        coords={
        "iT": np.arange(np.shape(lon_phi)[0]),
        "nS": np.arange(np.shape(lon_phi)[1]),
     },
 )

ds.to_netcdf("/home/jbaldwin/WorldBank/WIND_HAZARD/IBTRACS_LANDFALL_TRACKS/ibtracsv04r00_landfall_philippines.nc", mode = 'w')


# # Start from here if subset data already:

# In[27]:
# Load subset data of landfalling storms over Philippines
dat = xr.open_dataset('/home/jbaldwin/WorldBank/WIND_HAZARD/IBTRACS_LANDFALL_TRACKS/ibtracsv04r00_landfall_philippines.nc')
lon = np.array(dat.lon)
lat = np.array(dat.lat)
wspd = np.array(dat.wspd)/1.944 #convert from kts to m/s
days = np.array(dat.days)
year = np.array(dat.year)

# In[28]:
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


# In[29]:
# Interpolate to 15-min timesteps
nscale = 24 # convert from 6-hr to 15-min timesteps (factor of 6*4=24)
lon = ld.rescale_matrix(lon,nscale,0) # int for time interpolated 
lat = ld.rescale_matrix(lat,nscale,0)
wspd = ld.rescale_matrix(wspd,nscale,0)
days = ld.rescale_matrix(days,nscale,0)
tr = ld.rescale_matrix(tr,nscale,0)
trDir = ld.rescale_matrix(trDir,nscale,0)

# In[30]:
# Determine which storms make landfall in a rectangular region around the PHilippines and when those landfalls occur.
llon_midpoint = 180 # b/c in Western North Pacific not North Atlantic
nSlandfall_all_phi, iTlandfall_all_phi, nSlandfall_phi = landfall_in_box(lonmin,lonmax,latmin,latmax,lon,lat,wspd,llon_midpoint)

# In[32]:
# For each storm select time points 1 day before and after landfall, including possibility for second landfall and potential overlap. Used to determine timepoints to calculate windfields for.
days_before_landfall = 1
days_post_landfall = 1
timeres = 4*24 # timesteps per day
chaz = 0
iTlandfall_forwindfield_phi = timepoints_around_landfall( days_before_landfall, days_post_landfall, nSlandfall_all_phi, iTlandfall_all_phi, wspd, lon, tr, timeres, chaz)

# # In[50]:
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


# In[52]:


# Define x-y grid to put profile on
X = np.arange(116.5,127.6,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
Y = np.arange(4.5,19.1,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
X1, Y1 = np.meshgrid(X,Y)


# In[63]:


#CURRENT
missed_tries = 0
for nS in range(10):#range(len(wspd_landfall)):
    print(nS)
    
    # Select data for 1 storm
    wspd_nS = wspd_landfall[nS][:]
    lon_nS = lon_landfall[nS][:]
    lat_nS = lat_landfall[nS][:]
    days_nS = days_landfall[nS][:]
    tr_nS = tr_landfall[nS][:]
    trDir_nS = trDir_landfall[nS][:]
    
    # Calculate radius of maximum wind
    rmax_nS = utility.knaff15(wspd_nS*1.944, lat_nS)  #  wspd should be input in kts, outputs in km
    rmax_min = 20 # km
    rmax_nS[np.where(rmax_nS<rmax_min)] = rmax_min # 6-12-20: set places with radii less than 0 to 20 km to fix convergence
    
    # Calculate wind fields in parallel
    stormpoints = np.shape(wspd_nS)[0]
    try:
        wspdmaps = Parallel(n_jobs=16, prefer="threads")(delayed(windfield)(X1,Y1,lon_nS,lat_nS,wspd_nS,rmax_nS,tr_nS,trDir_nS,i) for i in range(stormpoints))    
        wspdmaps = np.abs(wspdmaps) # take absolute value for places asymmetry correction overpowers wind speed
        wspdmaps = np.expand_dims(wspdmaps,axis = 0)  
    
        swath = np.nanmax(wspdmaps, axis = 1) # Calculate swath over windfields; nanmax to ignore timepoints that don't have windfields
        
        # Create swath dataset
        ds = xr.Dataset(
         {"swath": (("nS", "lat", "lon"), swath),
          "days": (("nS","iT"), np.expand_dims(days_nS,axis=0)),
          "year": (("nS"), [year[nS]])},
             coords={
            "nS":np.array([nS]),
            "iT":np.arange(stormpoints),
            "lat": Y,
            "lon": X,
         },
         )
    
        #Write to netcdf
        direc = '/data2/jbaldwin/WINDFIELDS/IBTRACS/PHI/SWATHS/'
        filename = 'wspd_phi_swaths_maxasymcorrec_ibtracsv04r00_3-8-21.nc'
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


