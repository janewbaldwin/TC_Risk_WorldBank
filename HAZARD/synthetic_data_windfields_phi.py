#!/usr/bin/env python
# coding: utf-8

# # Run these three cells first:

# In[157]:


#reset


# In[158]:


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
from pygplib import readbst
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

# Root directory: change to where data downloaded to
root_dir = '/data2/jbaldwin/WCAS2023'

# In[159]:


# Bounding box for the Philippines
lonmin = 117.17427453
latmin = 5.58100332277
lonmax = 126.537423944
latmax = 18.5052273625


# In[160]:


# File number to examine
fileN = sys.argv[1] # string eg 006 (converted to string by sys)


# # Initial subsetting of landfalling TCs over Philippines:
# Left here for reference, but commented out since data on DesignSafe is already subset.

# In[161]:


# # Load data from file
# filename = '/data2/clee/ERAInterim/ERAInterim_wpc_'+fileN+'.nc'
# ds0 = xr.open_dataset(filename)
# lon = ds0.longitude[:]
# lat = ds0.latitude[:]
# wspd = ds0.Mwspd[:] # wind speed in knots
# days = ds0.time[:] # date in days, 6 hour time steps
# days = np.where(days==-54786., np.nan, days)
# year = ds0.year[:]

# # Interpolate to 15-min timesteps
# nscale = 24 # convert from 6-hr to 15-min timesteps (factor of 6*4=24)
# lon_int = ld.rescale_matrix(lon,nscale,0) # int for time interpolated 
# lat_int = ld.rescale_matrix(lat,nscale,0)
# wspd_int = ld.rescale_matrix(wspd,nscale,1)
# days_int = ld.rescale_matrix(days,nscale,0)


# # In[163]:
# # Calculate which storms and when make landfall in Philippines
# wspd_int_forlandfall = np.nanmax(wspd_int, axis = 0)
# llon_midpoint = 180
# nSlandfall_all_phi, iTlandfall_all_phi, nSlandfall_phi = landfall_in_box(lonmin,lonmax,latmin,latmax,lon_int,lat_int,wspd_int_forlandfall,llon_midpoint)

# # Select data only for storms that make landfall in the Philippines
# # (for normal timesteps and interpolated timestep data)
# lon_phi = lon[:,nSlandfall_phi]
# lat_phi = lat[:,nSlandfall_phi]
# wspd_phi = wspd[:,:,nSlandfall_phi]
# days_phi = days[:,nSlandfall_phi]
# year_phi = year[nSlandfall_phi]

# lon_int_phi = lon_int[:,nSlandfall_phi]
# lat_int_phi = lat_int[:,nSlandfall_phi]
# wspd_int_phi = wspd_int[:,:,nSlandfall_phi]
# days_int_phi = days_int[:,nSlandfall_phi]


# # In[168]:


# # Save out data of Philippines landfalling TCs
# ensN = wspd.ensembleNum.values # count of ensemble numbers

# ds = xr.Dataset(
#     {"lon": (("iT","nS"), lon_phi),
#      "lat": (("iT","nS"), lat_phi),
#      "wspd": (("ensembleNum","iT","nS"), wspd_phi), # maximum sustained wind speed in m/s
#      "days": (("iT","nS"), days_phi),
#      "year": (("nS"), year_phi)},
#         coords={
#         "ensembleNum": ensN,
#         "iT": np.arange(np.shape(lon_phi)[0]),
#         "nS": np.arange(np.shape(lon_phi)[1]),
#      },
#  )

# ds_int = xr.Dataset(
#     {"lon": (("iT","nS"), lon_int_phi),
#      "lat": (("iT","nS"), lat_int_phi),
#      "wspd": (("ensembleNum","iT","nS"), wspd_int_phi), # maximum sustained wind speed in m/s
#      "days": (("iT","nS"), days_int_phi),
#      "year": (("nS"), year_phi)},
#         coords={
#         "ensembleNum": ensN,
#         "iT": np.arange(np.shape(lon_int_phi)[0]),
#         "nS": np.arange(np.shape(lon_int_phi)[1]),
#      },
#  )

# ds.to_netcdf('/data2/jbaldwin/WINDFIELDS/CHAZ/ERAInterim_WPC/PHI_LANDFALL_TRACKS/ERAInterim_wpc_'+fileN+'_landfall_philippines.nc', mode = 'w')
# ds_int.to_netcdf('/data2/jbaldwin/WINDFIELDS/CHAZ/ERAInterim_WPC/PHI_LANDFALL_TRACKS/ERAInterim_wpc_'+fileN+'_landfall_philippines_15min.nc', mode = 'w')


# Start from here if subset data already: (eg this is where to start from if downloaded data from DesignSafe, which is already subset for Philippines landfalling!!)

#In[169]:
# Load subset data of landfalling storms over Philippines
dat = xr.open_dataset(root_dir+'/HAZARD/TC_TRACKS/CHAZ/ERAInterim_wpc_'+fileN+'_landfall_philippines.nc')
lon = np.array(dat.lon)
lat = np.array(dat.lat)
wspd = np.array(dat.wspd)/1.944 #convert from kts to m/s
days = np.array(dat.days)
year = np.array(dat.year)

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


# In[171]:


# Interpolate to 15-min timesteps
nscale = 24 # convert from 6-hr to 15-min timesteps (factor of 6*4=24)
lon = ld.rescale_matrix(lon,nscale,0) # int for time interpolated 
lat = ld.rescale_matrix(lat,nscale,0)
wspd = ld.rescale_matrix(wspd,nscale,1) # last input set to 1 not 0 because wind speed has ensemble members so extra axis
days = ld.rescale_matrix(days,nscale,0)
tr = ld.rescale_matrix(tr,nscale,0)
trDir = ld.rescale_matrix(trDir,nscale,0)

# In[172]:
#%%time
# Retrieve times of landfall for the WNP
wspd_forlandfall = np.nanmax(wspd, axis = 0)

# In[30]:
# Determine which storms make landfall in a rectangular region around the Philippines and when those landfalls occur.
llon_midpoint = 180 # b/c in Western North Pacific not North Atlantic
nSlandfall_all_phi, iTlandfall_all_phi, nSlandfall_phi = landfall_in_box(lonmin,lonmax,latmin,latmax,lon,lat,wspd_forlandfall,llon_midpoint)

# In[177]:
# In[32]:
# For each storm select time points 1 day before and after landfall, including possibility for second landfall and potential overlap. Used to determine timepoints to calculate windfields for.
days_before_landfall = 1
days_post_landfall = 1
timeres = 4*24 # timesteps per day
chaz = 1
iTlandfall_forwindfield_phi = timepoints_around_landfall( days_before_landfall, days_post_landfall, nSlandfall_all_phi, iTlandfall_all_phi, wspd_forlandfall, lon, tr, timeres, chaz)

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
    wspd_landfall.append(wspd[:,iT,nS]) #extra dimension = wspd includes 40 ensemble members for each track
    lon_landfall.append(lon[iT,nS])
    lat_landfall.append(lat[iT,nS])
    days_landfall.append(days[iT,nS])
    tr_landfall.append(tr[iT,nS])
    trDir_landfall.append(trDir[iT,nS])


# Define x-y grid to put profile on
X = np.arange(116.5,127.6,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
Y = np.arange(4.5,19.1,0.1) #Philippines lon rounded to nearest whole degree (down for min, up for max), plus 0.5deg further for wind radius
X1, Y1 = np.meshgrid(X,Y)


# In[257]:


#%%time

# Make sure to delete ens_swaths.nc and wspd_phi_swaths+fileN+.nc before running, otherwise won't pass through properly
direc = root_dir+'/HAZARD/WIND_SWATHS/CHAZ/'
print('Starting to run out'+filename+', putting results in'+direc+'.')
ensembleNum = dat.ensembleNum.values
missed_tries = 0
for nS in np.arange(0,len(wspd_landfall),1):
    
    # Info that is consistent across ensemble members
    lon_nS = lon_landfall[nS][:]
    lat_nS = lat_landfall[nS][:]
    days_nS = days_landfall[nS][:]
    tr_nS = tr_landfall[nS][:]
    trDir_nS = trDir_landfall[nS][:]
    swaths = np.zeros([1,len(ensembleNum),len(Y),len(X)])
    
    # Calculate wind swath for each ensemble member
    for ensN in ensembleNum:
    
        print(str(nS)+'-ens#'+str(ensN))
    
        # Find nan points
        notnans_wspd = ~np.isnan(wspd_landfall[nS][ensN,:]) # remove nan points from TC dissipating sooner than 1 day after first landfall
        if np.sum(notnans_wspd) == 0: # if no storm points leave wspdmaps[ensN,...] populated with zeros and move to next
            pass
        
        else:
            # Select data for 1 storm
            wspd_nS_ensN = wspd_landfall[nS][ensN,notnans_wspd]
            lon_nS_ensN = lon_landfall[nS][notnans_wspd]
            lat_nS_ensN = lat_landfall[nS][notnans_wspd]
            tr_nS_ensN = tr_landfall[nS][notnans_wspd]
            trDir_nS_ensN = trDir_landfall[nS][notnans_wspd]
    
            # Calculate radius of maximum wind
            rmax_nS_ensN = utility.knaff15(wspd_nS_ensN*1.944, lat_nS_ensN)  #  wspd should be input in kts, outputs in km
            rmax_min = 20 # minimum rmax used if Knaff15 produces values that are unrealistically small; km
            rmax_nS_ensN[np.where(rmax_nS_ensN<rmax_min)] = rmax_min # 6-12-20: set places with radii less than 0 to 20 km to fix convergence
    
            # Calculate wind fields in parallel
            stormpoints = np.shape(wspd_nS_ensN)[0]  
            try:
                wspdmaps = Parallel(n_jobs=3, prefer="threads")(delayed(windfield)(X1,Y1,lon_nS_ensN,lat_nS_ensN,wspd_nS_ensN,rmax_nS_ensN,tr_nS_ensN,trDir_nS_ensN,i) for i in range(stormpoints))    
                wspdmaps = np.abs(wspdmaps) # take absolute value for places asymmetry correction overpowers wind speed
                swaths[0,ensN,:,:] = np.nanmax(wspdmaps, axis = 0) # Calculate swath over windfields; nanmax to ignore timepoints that don't have windfields
            except:
                missed_tries += 1
                
    # Create swath dataset
    ds_ens = xr.Dataset(
        {"swath": (("nS", "ensembleNum", "lat", "lon"), swaths),
        "days": (("nS","iT"), np.expand_dims(days_nS,axis=0)),
        "year": (("nS"), [year[nS]])},
            coords={
        "nS": np.array([nS]),
        "ensembleNum": ensembleNum,
        "iT": np.arange(len(lon_nS)),
        "lat": Y,
        "lon": X,
        },
        )
            
    #Write each track with its many ensemble members to netcdf        
    filename = 'wspd_phi_swaths_'+fileN+'_'+str(nS).zfill(3)+'.nc'
    ds_ens.to_netcdf(direc+filename,mode='w',unlimited_dims = ["nS"])

print("My program took", time.time() - start_time, "to run and had ", missed_tries, " missed tries.")
