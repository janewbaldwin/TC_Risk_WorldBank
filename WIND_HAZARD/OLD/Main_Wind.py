#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
from datetime import datetime
#get_ipython().run_line_magic('pip', 'install netCDF4')
import netCDF4 


# In[2]:


### define function that reads netcdf data containing all global typhoon data
class read_ibtracs_v4(object):
      """
        a function read /data2/clee/bttracks/IBTrACS.ALL.v04r00.nc
        the data is every 3 hours, and thus in order to get 6 hourly data, you need to add 'gap' variable
        netCDF4 library is required
        we use all USA agency - usa_lat, usa_lon, etc 
        atl - NHC ATL
        enp - NHC ENP
        wnp,sh,ni - JTWC for the rest: SH, IO, and WPC
        'global', it is reading all data
        

        lon is change to the range from 0 to 360

        to do task: use xarray to have better time calculations
        data = xr.to_nedcdf(filename)
      """
      def __init__(self,ncFileName,basins,gap):
        #ncFileName = 'IBTrACS.ALL.v04r00.nc'
        nc = netCDF4.Dataset(ncFileName,'r', format='NETCDF4_CLASSIC')
    ### filters global typhoon data based on input 'basins', which come from the list of basins defined by NCDC 
    ### (atlantic, western pacific, easter pacific, north indian, southern hemisphere, global)
        sourceN = []
        if 'atl' in basins:
           sourceN.extend(['NA'])
        if 'wnp' in basins:
           sourceN.extend(['WP'])
        if 'sh' in basins:
           sourceN.extend(['SA','SP','SI'])
        if 'ni' in basins:
           sourceN.extend(['NI'])
        if 'enp' in basins:
           sourceN.extend(['EP'])
        if 'global' in basins:
           sourceN = ['NA','WP','SA','SP','SI','NI','EP']
        sourceN = np.array(sourceN)
        print (sourceN,basins)
    ### collates filtered typhoons into an array
        arg=[]
        for i in range(sourceN.shape[0]):
            arg.extend(np.argwhere((nc.variables['basin'][:,0,0]==sourceN[i][0].encode("utf-8"))&
                        (nc.variables['basin'][:,0,1]==sourceN[i][1].encode("utf-8"))).ravel().tolist())
        arg = np.array(arg)
        print(arg)
        
    ### from nc dataset, import variables. data is in 3hr intervals, gap allows to filter less frequent 
    ### data (3n-hourly) [gap=2 is used]
        lon = nc.variables['usa_lon'][:][arg,::gap].T
        lat = nc.variables['usa_lat'][:][arg,::gap].T
        wspd = nc.variables['usa_wind'][:][arg,::gap].T
        days = nc.variables['time'][:][arg,::gap].T
        names = nc.variables['name'][:][arg,::gap].T
        stormID = nc.variables['numobs'][arg].T 
        dist2land = nc.variables['dist2land'][:][arg,::gap].T
        trspeed = nc.variables['storm_speed'][:][arg,::gap].T
        trdir = nc.variables['storm_dir'][:][arg,::gap].T
        year = nc.variables['season'][arg].T
        times = nc.variables['iso_time'][:][arg,::gap,:].T
    
    ### define variables
        nNaN = np.argwhere(np.nanmax(np.array(lon),axis=0)!=-9999.).ravel()
        lon = lon[:,nNaN]
        lat = lat[:,nNaN]
        wspd = wspd[:,nNaN]
        days = days[:,nNaN]
        times = times[:,:,nNaN]
        names = names[:,nNaN]
        stormID = stormID[nNaN]
        dist2land = dist2land[:,nNaN]
        trspeed = trspeed[:,nNaN]
        trdir = trdir[:,nNaN]
        year = year[nNaN]

        a = np.nanmax(wspd,axis=0)
        arg = np.argwhere(a==a)[:,0]
        lon = np.array(lon[:,arg])
        lat = np.array(lat[:,arg])
        year = year[arg]
        wspd = np.float_(np.array(wspd[:,arg]))
        days = np.array(days[:,arg])
        dist2land = np.array(dist2land[:,arg])
        wspd[wspd==-9999.] = np.float('nan')
        lon[lon==-9999.] = np.float('nan')
        lat[lat==-9999.] = np.float('nan')
        stormID = np.array(stormID[arg])
        names = np.array(names[:,arg])
        trspeed = np.array(trspeed[:,arg])
        trdir = np.array(trdir[:,arg])
        times = np.array(times[:,:,arg])

        lon[lon<0]=lon[lon<0]+360
        self.wspd = wspd
        self.lon = lon
        self.lat = lat
        self.year = year
        self.days = days
        self.stormID = stormID
        self.names = names
        self.dist2land = dist2land
        self.year = year
        self.trspeed = trspeed
        self.trdir = trdir
        self.times = times
        self.r5 = np.zeros(year.shape)


# In[3]:


### define function that determines whether a cell is designated as land or not
from scipy.io import netcdf_file
def get_landmask(filename):
    """
    read 0.25degree landmask.nc 
    output:
    lon: 1D
    lat: 1D
    landmask:2D
  

    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = -f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask


# In[4]:


def rescale_array(array,scale):
    new_array = []
    for value in range(len(array)):
        if value != (len(array)-1):
           new_array.append(np.linspace(array[value], array[value +1], scale+1))
           
    return new_array

def rescale_array2(tc_lon,tc_lat,tc_speed,scale):
    """
    linear interpolate data into finer resolution
    scale: incred resolution in grid number
    """
 
    tc_lon = tc_lon[tc_lon == tc_lon]
    tc_lat = tc_lat[tc_lat == tc_lat]
    original_length = len(tc_lon)
    new_values_lon = [] 
    ## remove repeated values (every sixth value) after running function on arrays 
    extra_indices = [x*scale for x in range(1,original_length -1)]
    tc_lon = rescale_array(tc_lon,scale+1)
    tc_lon = np.array(tc_lon)
    tc_lon = np.delete(tc_lon, extra_indices)
    tc_lat = rescale_array(tc_lat,scale+1)
    tc_lat = np.array(tc_lat)
    tc_lat = np.delete(tc_lat, extra_indices)
    tc_lon = np.array(tc_lon)
    tc_lat = np.array(tc_lat)
    tc_speed = rescale_array(tc_speed,scale+1)
    tc_speed = np.array(tc_speed)
    tc_speed = np.delete(tc_speed, extra_indices)

    return tc_lon, tc_lat,tc_speed


# In[5]:


### filters storm information from iBTrACs data with relevant landmask and area to 
### generate index of storms and their corresponding landfall
def filterLandfalliS(basin, y1,y2,lon1,lon2,lat1,lat2,bt,llon,llat,ldmask):
    """
    input storm information from iBTrACs, landmask, area of interest, 
    return index of landfall storms in the area of interest
    """
### filters typhoons that fall within input parameters    
    # phillippines 118-128, 5-20
    #removed (np.array(bt.basin)==basin) for now
    iib = np.argwhere((np.array(bt.year[:])>=y1) & (np.array(bt.year[:])<=y2))
    #added ravel+tolist
    ix = np.argwhere(((llon>lon1) & (llon<lon2))).ravel()
    iy = np.argwhere(((llat>lat1) & (llat<lat2))).ravel()
    phildmask = ldmask[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
    phillon = llon[ix[0]:ix[-1]+1]
    phillat = llat[iy[0]:iy[-1]+1]
    wplon = bt.lon[:,iib][:,:,0]
    wplat = bt.lat[:,iib][:,:,0]
    #changed variable name
    wpspeed = bt.trspeed[:,iib][:,:,0]
    nx,ny = np.where((wplon>phillon[0])&(wplon<phillon[-1])&(wplat>phillat[-1])&(wplat<phillat[0]))
### check which storms made landfall to Phillipines
    phiiS=[]
    iLandfall=[]
    for iS in np.unique(ny):
      tc_lon, tc_lat, tc_speed= wplon[:,iS], wplat[:,iS], wpspeed[:,iS]
      #1: increase frequecy from 6 hours to hourly
      tc_lon, tc_lat, tc_speed = rescale_array2(tc_lon,tc_lat,tc_speed,6) #removed ld
      #2: assign land or water for each recoreded storm location 
      tc_landmask = np.zeros(tc_lat.shape)*np.float('nan')
      for ii_tc in range (0,tc_lon[tc_lon==tc_lon].shape[0],1):
          i = np.argmin(np.abs(phillon-tc_lon[ii_tc]))
          j = np.argmin(np.abs(phillat-tc_lat[ii_tc]))
          tc_landmask[ii_tc] = phildmask[j,i]
          if  tc_landmask[ii_tc] > 0:
              phiiS.append(iS)
              iLandfall.append(np.int(np.float(ii_tc)/6.))
              break;
    phiiS = iib[phiiS][:,0]
    return phiiS,iLandfall
### phiis - storm id
### iLandfall - nearest point of entry (time)


# In[6]:


### running the program to filter relevant storms
#ibtrackfiles = 'Allstorms_WMO_tracks_IBTRACS_V03r08_netcdf.mat'
bt = read_ibtracs_v4('/data2/clee/bttracks/IBTrACS.ALL.v04r00.nc','wnp',2)
landmaskfile = 'landmask.nc'
llon,llat,ldmask = get_landmask(landmaskfile)
liS,iLandfall = filterLandfalliS('wnp',2000,2014,118.,128.,5.,20.,bt,llon,llat,ldmask)


# In[7]:
