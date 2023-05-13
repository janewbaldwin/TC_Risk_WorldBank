#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
from netCDF4 import Dataset
from scipy.io import netcdf_file
from scipy import interpolate
from geopy.distance import geodesic

def get_landmask(filename):
	"""
	Written by Chia-Ying Lee (LDEO/Columbia).
	read 0.25degree landmask.nc 
	output:
	lon: 1D
	lat: 1D
	landmask:2D

	"""
	f = netcdf_file(filename)
	lon = f.variables['lon'][:]
	lat = f.variables['lat'][::-1]
	landmask = f.variables['landmask'][:,:]
	f.close()

	return lon, lat, landmask

def rescale_matrix(array,scale,axis1):
	"""
	Written by Chia-Ying Lee (LDEO/Columbia).
	rescale matrix through linear-interpolations a
	if it is an 1-d arry, axis1 set to 0
	"""
	newax = np.arange(array.shape[axis1]*scale)
	oldax = newax[0::scale]
	f = interpolate.interp1d(oldax,array,axis=axis1,\
		fill_value='extrapolate')
	newarray = f(newax)
	return newarray

def get_landfall_stormID(lon,lat,wspd,llon,llat,ldmask,land,ocean):
	"""
	Written by Chia-Ying Lee (LDEO/Columbia).
	get landfalling storms' ID. 
	- lon, lat, and wspd are the trackset with dimentions of (nt, nS). nt is the 
	track length while nS is the storn number
	- llon, llat, and ldmask are the landmask. 	
	llon and llat are 1-D array and ldmask has same dimenions as (llat, llon)
	- land and ocean are the land and ocean values from ldmask
	"""
	i_landfall = []
	iS = 0
	while iS < lon.shape[1]:
		#1: assign land or water for each recoreded storm location 
		dummy_lon,dummy_lat,dummy_wspd = lon[:,iS],lat[:,iS],wspd[:,iS]
		dummy_lat = dummy_lat[((dummy_lon==dummy_lon)&(dummy_wspd==dummy_wspd))]
		dummy_lon = dummy_lon[((dummy_lon==dummy_lon)&(dummy_wspd==dummy_wspd))]
		ilandmask = np.zeros(dummy_lon.shape)*np.float('nan')
		for iiS in range (0,dummy_lon.shape[0],1):
			i = np.argmin(np.abs(llon-dummy_lon[iiS]))
			j = np.argmin(np.abs(llat-dummy_lat[iiS]))
			ilandmask[iiS] = ldmask[j,i]
		#2: check if the landmask have both land and ocean points
		if ((ocean in ilandmask) and (land in ilandmask)):
			i_landfall.append(iS)
		iS += 1
	return(np.array(i_landfall))

def get_landfall_storm_time(iSlandfall,lon,lat,wspd,xlon,xlat,ldmask,land,ocean,nscale):
	"""
	Written by Chia-Ying Lee (LDEO/Columbia).
	using the landfall storms' ID (iSlandfall) from the function
	"""
	nSlandfall_first,nSlandfall_all,iTlandfall_first,iTlandfall_all = [],[],[],[]
	for iS in iSlandfall:
		index_enter_iS = []
		index_enter_land = []
		index_leave_iS = []
		index_leave_land = []
		#1: assign land or water for each recoreded storm location 
		dummy_lon,dummy_lat = lon[:,iS],lat[:,iS]
		#dummy_lat = dummy_lat[dummy_lon==dummy_lon]
		#dummy_lon = dummy_lon[dummy_lon==dummy_lon]
		dummy_it = np.argwhere(dummy_lon==dummy_lon).ravel()
		ilandmask = np.zeros(dummy_lon.shape)+5
		#for iiS in range (0,dummy_lon.shape[0],1):
		for iiS in dummy_it:
				i = np.argmin(np.abs(xlon-dummy_lon[iiS]))
				j = np.argmin(np.abs(xlat-dummy_lat[iiS]))
				ilandmask[iiS] = ldmask[j,i]
		for ii in range(1,ilandmask.shape[0]):
			if ilandmask[ii]-ilandmask[ii-1] == land-ocean:
				index_enter_land.append(ii)
				index_enter_iS.append(iS)
			elif ilandmask[ii]-ilandmask[ii-1] == ocean-land:
				index_leave_land.append(ii)
				index_leave_iS.append(iS)
		if len(index_leave_land) > len(index_enter_land):
			index_leave_land = index_leave_land[1:]
		if len(index_enter_land)>0:
			datalonleave0 = dummy_lon[index_leave_land]
			datalatleave0 = dummy_lat[index_leave_land]
			datalonlatleave0 = np.column_stack((datalonleave0,datalatleave0))
			datalon = dummy_lon[index_enter_land]
			datalat = dummy_lat[index_enter_land]
			datalonlat = np.column_stack((datalon,datalat))

			#### first landfall:
			iTlandfall_first.append(index_enter_land[0])
			nSlandfall_first.append(index_enter_iS[0])
			iTlandfall_all.append(index_enter_land[0])
			nSlandfall_all.append(index_enter_iS[0])
			#### landfall points with criteria
			if len(index_enter_land)>1:
				for x in range(1,len(index_leave_land)):
					difDis = geodesic((datalonlat[x,1],datalonlat[x,0]),(datalonlat[x-1,1],datalonlat[x-1,0])).kilometers
					difTime = (index_leave_land[x]-index_enter_land[x])*6./(1.0*nscale) # hours in float
					#### if leaving land points are 80km larger than the previous landfall points, keep the indecies
					#### of if the leaving point is 6 hours from the next landfall points
					if ((difDis >= 80) or (difTime >= 24)):
						iTlandfall_all.append(index_enter_land[x])
						nSlandfall_all.append(index_enter_iS[x])
				#### last landfall points:
				if len(index_enter_land) > len(index_leave_land):
					#print (iS)
					if len(index_enter_land) == 2:
						x = 1
					else:
						x = x+1
					iTlandfall_all.append(index_enter_land[x]) 	
					nSlandfall_all.append(index_enter_iS[x])
					
		#else:
		#	#print (iS)
		#	#plt.plot(dummy_lon,dummy_lat)
	return(np.array(nSlandfall_first),np.array(iTlandfall_first),np.array(nSlandfall_all),np.array(iTlandfall_all))



