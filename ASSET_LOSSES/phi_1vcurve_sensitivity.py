#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe
import julian
import datetime
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time

# For timing the script
start_time = time.time()

#Bounding Box for Philippines
lonmin = 117.17427453
latmin = 5.58100332277
lonmax = 126.537423944
latmax = 18.5052273625

# function for Emanuel Vulnerability Curves

def vulnerability(V,Vthresh,Vhalf): # calculate fractional property value lost over space

    # V = Input wind speed swath; m/s
    
    # Vthresh = windspeed at and below which no damage occurs; m/s

    # Vhalf = windspeed at which half the property value is lost; m/s

    vn0 = V-Vthresh
    vn = np.maximum(vn0, np.zeros(np.shape(vn0)))/(Vhalf-Vthresh)
    f = vn**3/(1+vn**3)
    
    return f # vulnerability in all different regions

# # Open up swaths:

ds_swaths = xr.open_dataset('/data2/jbaldwin/WINDFIELDS/IBTRACS/PHI/SWATHS/wspd_phi_swaths_maxasymcorrec_ibtracsv04r00_3-8-21.nc')
swath = ds_swaths.swath

# Convert modified Julian days to date-time objects
nmax = len(ds_swaths.nS)
imax = len(ds_swaths.iT)
dt = np.full([nmax, imax], np.nan, dtype='datetime64[s]')
month = np.full([nmax, imax], np.nan)
day = np.full([nmax, imax], np.nan)
for n in range(nmax):
    mjd = ds_swaths.days[n,:]
    for i in np.where(~np.isnan(mjd))[0]:
        x = julian.from_jd(mjd[i], fmt='mjd').date()
        dt[n][i] = x
        month[n,i] = int(x.month)
        day[n,i] = int(x.day)
year = ds_swaths.year.values

# Calculate TC start and end dates
tc_start_date = np.min(dt,axis=1)
tc_end_date = np.max(dt,axis=1)

# Exposed Value Data, subset over Philippines
ds_exp = xr.open_dataset('/data2/jbaldwin/EXPOSED_VALUE/LITPOP/litpop_v1-2_phl.nc').sel(lon=slice(lonmin,lonmax),lat=slice(latmin,latmax))
exposed_value = ds_exp.value

# Determine Regridder for Hazard --> Exposed Value
regridder = xe.Regridder(swath, ds_exp, 'bilinear',reuse_weights=True)

# Regrid wind swath
swath_out = regridder(swath)

# Calculate asset losses for various Vhalf and Vthresh values

Vthresh = np.arange(15,40,5)# Eberenz et al 2020, 25.7 is value used for all and 10 below and above that; m/s
Vhalf = np.arange(50,210,10)# Eberenz et al 2020, value for Philippines using default, RMSF calculation, and TDR calculation; m/s
V = swath_out
asset_losses = np.full(np.shape(swath_out),np.nan)

for vt in Vthresh:
    for vh in Vhalf:
        for nS in range(nmax):
            print('Vthresh = '+str(vt)+', Vhalf='+str(vh)+', nS = '+str(nS))
            f = vulnerability(V.sel(nS=nS).values,vt,vh)
            asset_losses[nS,...] = exposed_value*f

        # Save out dataset of asset losses with dates
        ds_asset_losses = xr.Dataset(
                 {"asset_losses": (("nS","lat", "lon"), asset_losses),
                 "start_date": (("nS"), tc_start_date),
                 "end_date": (("nS"), tc_end_date)},
                    coords={
                    "nS":V.nS,
                    "lat":V.lat,
                    "lon":V.lon,
                 },
                 )

        ds_asset_losses.to_netcdf('/data2/jbaldwin/WINDFIELDS/IBTRACS/PHI/ASSET_LOSSES/VCURVE_SENSITIVITY/Vhalf-'+str(vh)+'_Vthresh-'+str(vt)+'.nc',mode='w')

        
# Report on timing        
print("My program took", time.time() - start_time, "to run.")



