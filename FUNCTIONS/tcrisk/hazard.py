# STANDARD PACKAGES
import numpy as np
import math
from past.utils import old_div
# SPECIALIZED CHAZ PACKAGES
from wind_reconstruct.w_profile_2 import W_profile
from chaz.utility import translationspeedFactor
from pygplib3 import landfall as ld

def distancefrompoint(lon, lat, X1, Y1):
    """
    Calculate distance between particular lat-lon point (eg center point of storm) and each point of lat-lon grid using Haversine formulation.    
    Adopted from: https://kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python
    :param lat: latitude location in degrees of point.
    :param lon: longitude location in degrees of point.
    :param X1: 2-D matrix of longitudes in degrees to put windfield on.
    :param Y1: 2-D matrix of latitudes in degrees to put windfield on.
    :return distance: 2-D matrix of distances in km.  
    """
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


def windfield_sym(X1,Y1,lon_nS,lat_nS,wspd_nS,rmax_nS,i):
    """
    Calculate wind field without adding in asymmetry.
    
    :param X1: 2-D matrix of longitudes in degrees to put windfield on.
    :param Y1: 2-D matrix of latitudes in degrees to put windfield on.
    :param lon_nS: vector of longitudes along TC track in degrees.
    :param lat_nS: vector of latitudes along TC track in degrees.
    :param wspd_nS: vector of wind speeds along TC track in m/s.
    :param rmax_nS: vector of radii of maximum wind along TC track in km.
    :param i: index of time point in TC track to calculate wind field for.
    :return wspdmap: map of wind field at one point in time in m/s. 
    """
    loni = lon_nS[i]
    lati = lat_nS[i]
    wspdi = wspd_nS[i]
    rmaxi = rmax_nS[i]
    
    # Calculate Willoughby Profile
    radius_max = 500
    radius_precision = 1
    profile = W_profile(lati, rmaxi, wspdi, radius_max, radius_precision)
    radius = np.arange(0,radius_max + radius_precision, radius_precision)
    
    # Create dict look-up table from Willoughby Profile
    wspdlookup = dict(zip(radius, profile))
    
    # Calculate distance from center of storm
    distance = distancefrompoint(loni, lati, X1, Y1) # distance in km
    
    # Round distance values to nearest whole number
    distance = distance.astype(int)

    # Remap radii to windspeed
    wspdmap = np.zeros(np.shape(distance))
    for r in radius:
        wspdmap[np.where(distance == r)] = wspdlookup[r]
        wspdmap[np.where(distance > radius_max)] = 0 # added 10-27-20
    
    return wspdmap


def windfield_orig(X1, Y1, lon_nS,lat_nS,wspd_nS,rmax_nS,tr_nS,trDir_nS,i):
    """
    Calculate wind field with asymmetry, but not subtracting max asymmetry from windspeed input to profile (so windfield values sometimes exceed maximum sustained windspeed).

    :param X1: 2-D matrix of longitudes in degrees to put windfield on.
    :param Y1: 2-D matrix of latitudes in degrees to put windfield on.
    :param lon_nS: vector of longitudes along TC track in degrees.
    :param lat_nS: vector of latitudes along TC track in degrees.
    :param wspd_nS: vector of wind speeds along TC track in m/s.
    :param rmax_nS: vector of radii of maximum wind along TC track in km.
    :param i: index of time point in TC track to calculate wind field for.
    :return wspdmap: map of wind field at one point in time in m/s. 
    """
    loni = lon_nS[i]
    lati = lat_nS[i]
    wspdi = wspd_nS[i]
    rmaxi = rmax_nS[i]
    tri = tr_nS[i]
    trDiri = trDir_nS[i]
    
    # Calculate tangential wind
    angle = np.arctan2((Y1-lati),(X1-loni)) - trDiri # define angle relative to track direction 
    vt = -tri*np.cos(np.pi/2 - angle) # calculate tangential wind; remove minus if southern hemisphere
    
    # Calculate Willoughby Profile
    radius_max = 500
    radius_precision = 1
    profile = W_profile(lati, rmaxi, wspdi, radius_max, radius_precision)
    radius = np.arange(0,radius_max + radius_precision, radius_precision)
    
    # Create dict look-up table from Willoughby Profile
    wspdlookup = dict(zip(radius, profile))
    
    # Calculate distance from center of storm
    distance = distancefrompoint(loni, lati, X1, Y1) # distance in km
    
    # Round distance values to nearest whole number
    distance = distance.astype(int)
    
    # Calculate rFactor to modulate track correction
    rFactor = translationspeedFactor(old_div(distance,rmaxi))

    # Remap radii to windspeed
    wspdmap = np.zeros(np.shape(distance))
    for r in radius:
        wspdmap[np.where(distance == r)] = wspdlookup[r]
    
    #Add track direction correction
    wspdmap = wspdmap+(rFactor*vt)
    
    # Set to 0 outside radius_max
    wspdmap[np.where(distance > radius_max)] = 0 # added 10-27-20
     
    return wspdmap


def windfield(X1, Y1, lon_nS,lat_nS,wspd_nS,rmax_nS,tr_nS,trDir_nS,i):
    """
    Calculate wind field with asymmetry, subtracting max asymmetry correction from maximum sustained windspeed  before calculating wind profile, so that wind field values do not exceed maximum sustained wind speed.
    
    :param X1: 2-D matrix of longitudes in degrees to put windfield on.
    :param Y1: 2-D matrix of latitudes in degrees to put windfield on.
    :param lon_nS: vector of longitudes along TC track in degrees.
    :param lat_nS: vector of latitudes along TC track in degrees.
    :param wspd_nS: vector of wind speeds along TC track in m/s.
    :param rmax_nS: vector of radii of maximum wind along TC track in km.
    :param i: index of time point in TC track to calculate wind field for.
    :return wspdmap: map of wind field at one point in time in m/s. 
    """
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
    rFactor = translationspeedFactor(old_div(distance,rmaxi))
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


def landfall_in_box(lonmin,lonmax,latmin,latmax,lon,lat,wspd,llon_midpoint):
    """
    Determine which storms make landfall in a rectangular region and when those landfalls occur.
    Typically input lon, lat, wspd are interpolated to 15 minute temporal resolution before being input to this function.
    
    :param lonmin: Minimum longitude bound of region in degrees.
    :param lonmax: Maximum longitude bound of region in degrees.
    :param latmin: Minimum latitude bound of region in degrees.
    :param latmax: Maximum latitude bound of region in degrees.
    :param lon: 2-D matrix of longitudes of TC tracks (0th dim = time points, 1st dim = storm number).
    :param lat: 2-D matrix of latitudes of TC tracks (0th dim = time points, 1st dim = storm number).
    :param wspd: 2-D matrix of maximum wind speeds of TC tracks (0th dim = time points, 1st dim = storm number).
    :param llon_midpoint: Midpoint of the longitude range for the land mask. Set to 180 for WNP (0 to 360) and 0 for NA (-180 to 180).
    :return nSlandfall_all_box: List of indices of storms that make landfall in region at each iT time point. A particular nS will repeat for each landfall that occurs in region.
    :return iTlandfall_all_box: List of indices of time points when storms indicated in nSlandfall_all_box make landfall. 
    :return nSlandfall_box: List of indices of storms that make landfall in region at each iT time point. Even with multiple landfalls each storm index is only listed once.
    """
    # Load land-sea mask
    llon, llat, ldmask = ld.get_landmask('/home/clee/CHAZ/landmask.nc')
    llon2 = np.copy(llon)
    if llon_midpoint == 0:
        llon2[np.where(llon2>180)] -= 360 # correction so -180 to 180 rather than 0 to 360, and don't get weird jumps at lon=0
    land = np.max(ldmask)
    ocean = np.min(ldmask)
    
    # Retrieve times of landfall
    iSlandfall = ld.get_landfall_stormID(lon, lat, wspd, llon2, llat, ldmask, land, np.min(ldmask))
    landfall_times = ld.get_landfall_storm_time(iSlandfall, lon, lat, wspd, llon2, llat, ldmask, land, ocean, 24)
    nSlandfall_first = landfall_times[0] # index of storms that make first landfall
    iTlandfall_first = landfall_times[1] # time of making first landfall
    nSlandfall_all = landfall_times[2] # index of storms that make any landfall (ie storm would repeat if makes multiple landfalls)
    iTlandfall_all = landfall_times[3] # time of making that landfall
    
    # Select storms making landfall in Philippines
    nSlandfall_all_box = []
    iTlandfall_all_box = []
    for j in range(np.shape(nSlandfall_all)[0]):
        nS = nSlandfall_all[j]
        iT = iTlandfall_all[j]
        lon_landfall = lon[iT,nS]
        lat_landfall = lat[iT,nS]
        if lonmin <= lon_landfall <= lonmax and latmin <= lat_landfall <= latmax:
            nSlandfall_all_box.append(nSlandfall_all[j])
            iTlandfall_all_box.append(iTlandfall_all[j])

    # Remove duplicate storms (storms that made landfall in region twice)
    nSlandfall_box = list(dict.fromkeys(nSlandfall_all_box))
    
    return nSlandfall_all_box, iTlandfall_all_box, nSlandfall_box


def timepoints_around_landfall( days_before_landfall, days_post_landfall, nSlandfall_all_box, iTlandfall_all_box, wspd, lon, tr, timeres, chaz):
    """
    For each storm select time points some number of days before and after landfall, including possibility for second landfall and potential overlap. Used to determine timepoints to calculate windfields for. Typically run in sequence with landfall_in_box function. If using with CHAZ data with multiple ensembles, chaz = 1. For IBTrACS data, chaz = 0.

    :param days_before_landfall: Number of days to include before landfall.
    :param days_post_landfall: Number of days to include after landfall.
    :param nSlandfall_all_box: List of indices of storms that make landfall in region at each iT time point. A particular nS will repeat for each landfall that occurs in region.
    :param iTlandfall_all_box: List of indices of time points when storms indicated in nSlandfall_all_box make landfall.
    :param wspd: 2-D matrix of maximum wind speeds of TC tracks (0th dim = time points, 1st dim = storm number). Used to remove time points that wind speeds aren't available for.
    :param lon: 2-D matrix of longitudes of TC tracks (0th dim = time points, 1st dim = storm number). Used to remove time points that wind speeds aren't available for in the CHAZ data.
    :param tr: 2-D matrix of track speeds of TC tracks (0th dim = time points, 1st dim = storm number). Used to remove time points that track speeds aren't available for.
    :param timeres: scalar with number of time points of the TC track per day (4*24 for 15-min resolution).
    :param chaz: binary 1/0 where 1 indicates this is chaz data with multiple ensembles, whereas 0 is for use with IBTrACS.
    :return iTlandfall_forwindfield_box: Indices of time points around landfall for each storm.
    """
    timesteps_before_landfall = days_before_landfall*timeres
    timesteps_post_landfall = days_post_landfall*timeres 

    iTlandfall_forwindfield_box = []
    nSlandfall_forwindfield_box = []
    for i in range(np.shape(wspd)[1]):
        nSlandfall_forwindfield_box.append(i)
        indices_landfalls = np.where(np.array(nSlandfall_all_box)==i)[0] # different landfalls per storm
        nlandfalls = len(indices_landfalls)
        if nlandfalls == 1:
            iTlandfall = iTlandfall_all_box[indices_landfalls[0]]
            iT = np.arange(iTlandfall-timesteps_before_landfall,iTlandfall+timesteps_post_landfall+1,1)
            iTlandfall_forwindfield_box.append(list(iT))
        if nlandfalls > 1:
            iTs = np.array([],dtype=int)
            for ii in indices_landfalls:
                iTlandfall = iTlandfall_all_box[ii]
                iT = np.arange(iTlandfall-timesteps_before_landfall,iTlandfall+timesteps_post_landfall+1,1)
                iTs = np.concatenate((iTs,iT),axis=0)
            iTs = np.unique(iTs) # remove wind field points that repeat
            iTlandfall_forwindfield_box.append(list(iTs))
        iTlandfall_forwindfield_box[i] = list(filter(lambda x : x > 0, iTlandfall_forwindfield_box[i])) # remove negative numbers from list https://www.geeksforgeeks.org/python-remove-negative-elements-in-list/ 
        iTlandfall_forwindfield_box[i] = list(filter(lambda x : x <= np.max(np.where(~np.isnan(tr[:,i]))), iTlandfall_forwindfield_box[i])) # remove numbers above max time step for each storm based on tr which has fewer points than wspd,lat,lon
        if chaz == 1:
                iTlandfall_forwindfield_box[i] = [iT for iT in iTlandfall_forwindfield_box[i] if iT not in np.where(np.isnan(lon[:,i]))[0]] # remove any other nan values (this is done based on wspd for ibtracs data, but can't be here due to ensemble members)
        else:
                iTlandfall_forwindfield_box[i] = [iT for iT in iTlandfall_forwindfield_box[i] if iT not in np.where(np.isnan(wspd[:,i]))[0]] # remove any remaining nan values (wspd often has nan values at beginning, or sometimes in the middle)
        
    return iTlandfall_forwindfield_box
