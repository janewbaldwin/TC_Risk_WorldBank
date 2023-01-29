#/usr/bin/env python
import numpy as np
import copy
from netCDF4 import Dataset
from datetime import datetime,timedelta

class read_ibtracs_v4(object):
	"""
	a function read /data2/clee/bttracks/IBTrACS.ALL.v04r00.nc
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
		#ncFileName = '/data2/clee/bttracks/IBTrACS.ALL.v04r00.nc'
		nc = Dataset(ncFileName,'r', format='NETCDF3_CLASSIC')
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
		arg=[]
		for i in range(sourceN.shape[0]):
			arg.extend(np.argwhere((nc.variables['basin'][:,0,0]==sourceN[i][0].encode("utf-8"))&
			(nc.variables['basin'][:,0,1]==sourceN[i][1].encode("utf-8"))).ravel().tolist())
			arg = np.array(arg)
		#print(arg)
		lon = nc.variables['usa_lon'][:][arg,::gap].T
		lat = nc.variables['usa_lat'][:][arg,::gap].T
		wspd = nc.variables['usa_wind'][:][arg,::gap].T
		days = nc.variables['time'][:][arg,::gap].T
		names = nc.variables['name'][:][arg,:].T 
		stormID = nc.variables['number'][:][arg] 
		dist2land = nc.variables['dist2land'][:][arg,::gap].T
		trspeed = nc.variables['storm_speed'][:][arg,::gap].T
		trdir = nc.variables['storm_dir'][:][arg,::gap].T
		year = nc.variables['season'][:][arg]
		times = nc.variables['iso_time'][:][arg,::gap].T

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

