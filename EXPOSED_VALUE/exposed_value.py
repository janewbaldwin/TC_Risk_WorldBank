# conda activate worldbank


# Import packages
import geopandas as gpd
import matplotlib.pyplot as plt
import csv
import numpy as np
import xarray as xr
import regionmask
from shapely.geometry import Polygon
import time
from datetime import timedelta

#Determine start time
start_time = time.monotonic()

# Load country polygons
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Calculate GDP per capita
world = world[(world.pop_est>0) & (world.name!="Antarctica")]
world['gdp_per_cap'] = world.gdp_md_est / world.pop_est


# Read capital data csv file
csv_file = '/home/jbaldwin/WorldBank/AVGRET/capital_data.csv'
with open(csv_file, mode='r') as csv_file:
    readCSV = csv.reader(csv_file)
    code = []
    year = []
    cgdpo = []
    ck = []
    for column in readCSV:
        code0 = column[0]
        year0 = column[1]
        cgdpo0 = column[2]
        ck0 = column[3]
        
        code.append(code0)
        year.append(year0)
        cgdpo.append(cgdpo0)
        ck.append(ck0)
    
code.remove('code')
year.remove('year')
cgdpo.remove('cgdpo')
ck.remove('ck')


# Calculate average return on capital for different countries
avgret = []
for i in range(len(code)):
    avgret.append(float(cgdpo[i])/float(ck[i]))
    
avgret[163] = None #value for Zimbabwe = 19, which is way too high


# Make list of average return on capital 
avgret2 = []
for key in world.iso_a3:
    if key in code:
        i = code.index(key)
        avgret2.append(avgret[i])
    if key not in code:
        avgret2.append(None)

world['avg_ret'] = avgret2


# Calculate GDP per capital divided by average return
world['gdp_per_cap_div_avg_ret'] = world.gdp_per_cap / world.avg_ret


# Load population dataset
#pop_dat = xr.open_dataset('/home/jbaldwin/WorldBank/zsurf.nc')
pop_dat = xr.open_dataset('/data2/jbaldwin/EXPOSED_VALUE/WORLDPOP/ppp_2020_1km_Aggregated.nc')


# Rotate longitude to be 0-360
pop_dat = pop_dat.assign_coords(lon=(pop_dat.lon % 360)).roll(lon=21600, roll_coords=True)


# Select lat-lon-population variables and lat-lon bounds from population dataset
#lon = pop_dat.GRID_XT
#lat = pop_dat.GRID_YT
#pop = pop_dat.ZSURF
lon = pop_dat.lon
lat = pop_dat.lat
pop = pop_dat.Band1
LON_EDGE = lon
LAT_EDGE = lat


# Create region set of countries with their numbers
numbers = np.arange(0,world.shape[0],1).tolist()
names = world.name.tolist()
abbrevs = world.iso_a3.tolist()
polys = world.geometry.tolist()
countries_poly = regionmask.Regions_cls('countries', numbers, names, abbrevs, polys)


# Create country mask at resolution of population dataset (NOTE: TAKES A LONG TIME)
mask = countries_poly.mask(lon, lat, wrap_lon=True)
mask_ma = np.ma.masked_invalid(mask)


# Assign values [GDP per capita/ average return] to country mask
values = world.gdp_per_cap_div_avg_ret.tolist()
mask_ma_val = mask_ma
for i in numbers:
        mask_ma_val[np.where(mask_ma == i)] = values[i]
        
        
# Multiply value country mask by population dataset to get exposed value
pop_mask = pop*mask_ma_val


# Save out netcdf of exposed value
ds = xr.Dataset({'exposed_value': (('lat', 'lon'), pop_mask)},
                coords={'lon': np.array(lon), 'lat': np.array(lat)})
ds.to_netcdf('/data2/jbaldwin/EXPOSED_VALUE/exposed_value.nc')

# Print end time
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
