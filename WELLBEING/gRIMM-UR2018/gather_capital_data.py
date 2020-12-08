from res_ind_lib import *

from sorted_nicely import *
from replace_with_warning import *

import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)
from wb_api_wraper import *
import numpy as np
import pandas as pd

#Pandas display options
# pd.set_option('display.max_colwidth', 200)
# pd.set_option('display.width', 200)
# pd.set_option('display.precision', 10)
# pd.set_option('display.max_rows', 500)

#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
#%matplotlib inline
font = {'family' : 'serif',
    'weight' : 'normal',
    'size'   : 15}
plt.rc('font', **font)

from fancy_plots import *

#Names to WB names
any_to_wb = pd.read_csv("inputs/any_name_to_wb_name.csv",index_col="any",squeeze=True)

#GAR names with SIDS spec
gar_name_sids = pd.read_csv("inputs/gar_name_sids.csv")

#iso3 to wb country name table
iso3_to_wb=pd.read_csv("inputs/iso3_to_wb_name.csv").set_index("iso3").squeeze()

#iso2 to iso3 table
iso2_iso3 = pd.read_csv("inputs/names_to_iso.csv", usecols=["iso2","iso3"]).drop_duplicates().set_index("iso2").squeeze() #the tables has more lines than countries to account for several ways of writing country names

gar_name_sids['wbcountry'] = gar_name_sids.reset_index().country.replace(any_to_wb)

list_of_sids = gar_name_sids[gar_name_sids.isaSID=="SIDS"].dropna().reset_index().wbcountry

penn = pd.read_excel("inputs/pwt90.xlsx","Data")

def mrv_gp_2(x):
    """this function gets the most recent value from a dataframe grouped by country"""
    out= x.ix[(x["year"])==np.max(x["year"]),:]
    return out

hop=penn.groupby("country").apply(mrv_gp_2)
hop = hop.drop("country",axis=1).reset_index().drop("level_1",axis=1)

clean_penn = hop[['countrycode', 'country', 'year', 'cgdpo', 'ck']].copy()
clean_penn.head()

clean_penn["country"] = clean_penn.country.replace(any_to_wb)
GAR = pd.read_csv("inputs/GAR_capital.csv")
GAR["country"] = GAR.country.replace(any_to_wb)
gar_sids = GAR.set_index("country").loc[list_of_sids,:].replace(0, np.nan).dropna()
all_K = pd.concat([gar_sids,clean_penn.set_index("country")],axis=1)
all_K = all_K.loc[(all_K != 0).any(axis=1),:]
all_K.columns

all_K["prod_k_1"] = all_K.GDP/all_K.K
all_K["prod_k_2"] = all_K.cgdpo/all_K.ck
all_K.prod_k_2.describe()

all_K[all_K.prod_k_2>0.34]

all_K["avg_prod_k"] = all_K.prod_k_1
all_K.avg_prod_k = all_K.avg_prod_k.fillna(all_K.prod_k_2)

all_K["Y"] = all_K.cgdpo
all_K.Y = all_K.Y.fillna(all_K.GDP)

all_K["Ktot"] = all_K.ck
all_K.Ktot = all_K.Ktot.fillna(all_K.K)

all_K[["avg_prod_k"]].dropna().to_csv("intermediate/avg_prod_k_with_gar_for_sids.csv")
all_K.loc[list_of_sids,"avg_prod_k"].dropna()
