#!/usr/bin/env python
import numpy as np
from pygplib3 import readbst,util
from netCDF4 import Dataset
import pygplib3.landfall as ld
import xarray as xr
import scipy
from scipy.stats import genextreme as gev
from scipy import stats
from scipy.io import netcdf_file, loadmat

landmaskfile_50 = '/home/clee/WillisRe/landmask_50km.nc'
xlong,xlat,coastline = ld.landmask2coastline(landmaskfile_50)
nc = Dataset(landmaskfile_50,'r',format='NETCDF3_CLASSIC')
land = np.array(nc.variables['land'][0,:,:])
xlong = nc.variables['lon'][:]
xlat = nc.variables['lat'][:]
land = np.squeeze(land)
model,iens,tcgi = 'CCSM4','1',''
filename = '/home/clee/SwissRe/biascorrection/'+model+'_ATL_2005_2ens'+util.int2str(iens,3)+tcgi+'.nc'
f = xr.open_dataset(filename)
### to save time finding landfall storm and timeing
### we derived the maximum wind (over the intensity ensemble) at each time step
nscale = 24 # every 15 minutes
vmax = np.nanmax(f.Mwspd,axis=0)
chazlon1 = ld.rescale_matrix(f.longitude.values,nscale,0)
chazlat1 = ld.rescale_matrix(f.latitude.values,nscale,0)
chazwspd1 = ld.rescale_matrix(vmax,nscale,0)
### get the landfall storm ID. it is necessary to use 
### low-res landmask and high-frequency track data 
### to have highest coverage
iSlandfall = \
        ld.get_landfall_stormID(chazlon1,chazlat1,chazwspd1,\
        xlong,xlat,land,0,-1)
### get time of landfall. nS1 and tS1 are for the 
### first landfall while 
### nS2 adn tS2 are for all landfall points
nS1,tS1,nS2,tS2 = \
        ld.get_landfall_storm_time(iSlandfall,chazlon1,\
        chazlat1,chazwspd1,xlong,xlat,land,0,-1,nscale)
