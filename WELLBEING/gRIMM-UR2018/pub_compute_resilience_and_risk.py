#%reset-f
#%load_ext autoreload
#%autoreload 2

from res_ind_lib import *

from sorted_nicely import *
from replace_with_warning import *

import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)

import numpy as np
import pandas as pd

#Pandas display options
# pd.set_option('display.max_colwidth', 200)
# pd.set_option('display.width', 200)
# pd.set_option('display.precision', 10)
# pd.set_option('display.max_rows', 500)

import matplotlib.pylab as plt
#%matplotlib inline
font = {'family' : 'serif',
    'weight' : 'normal',
    'size'   : 15}
plt.rc('font', **font)

economy = "country" #province, deparmtent
event_level = [economy, "hazard", "rp"]

macro = pd.read_csv("orig_intermediate/macro.csv", index_col=economy).dropna()
macro.sample(n=2)

cat_info = pd.read_csv("orig_intermediate/cat_info.csv",  index_col=[economy, "income_cat"]).dropna()
cat_info.sample(n=2)

hazard_ratios = pd.read_csv("orig_intermediate/hazard_ratios.csv", index_col=event_level+["income_cat"]).dropna()
hazard_ratios.sample(n=2)

groups =  pd.read_csv("orig_inputs/income_groups.csv",header =4,index_col=2)

country_per_gp = groups["Income group"].reset_index().dropna().set_index("Income group").squeeze()
country_per_rg = groups["Region"].reset_index().dropna().set_index("Region").squeeze()
country_per_rg;

# args = dict(hazard_ratios = None)
args = dict(return_stats=True,hazard_ratios = hazard_ratios)
# args = dict(hazard_ratios = hazard_ratios.swaplevel("country","hazard").ix["earthquake"])
#args = dict()

#Computes
results, iah=compute_resilience(macro,cat_info,None,return_iah=True,optionPDS='unif_poor',optionB='data',verbose_replace=True,**args)

#Saves
results.to_csv("results/results.csv",float_format="%.9f")

#Quick statistics
print("nb countries with macro data :" +str(macro.shape[0]))
print("nb countries with cat data :"   +str(cat_info.unstack().dropna().shape[0]))
print("nb countries with hazard data :"+str(hazard_ratios.unstack(["rp","hazard", "income_cat"]).dropna().shape[0]))

nb_countries_all_results = results["resilience"].dropna().shape[0]
print("nb countries with results :"+str(nb_countries_all_results))


def print_stats(results, region_name="global"):

    #Some stats
    a=results.resilience;
    print("Resilience averages {mean:.0%} across our sample, ranging from {min:.0%} ({m}) to {max:.0%} ({M})".format(
            min=a.min(),mean=a.mean(),max=a.max(), m=a.argmin(),M=a.argmax()))

    a=results.risk
    print("Risk to welfare averages {mean:.2%} across our sample, ranging from {min:.03%} ({m}) to {max:.1%} ({M})".format(
        min=a.min(),mean=a.mean(),max=a.max(),m=a.argmin(),M=a.argmax()))

    a=results.risk_to_assets
    print("Risk to assets averages {mean:.2%} across our sample, ranging from {min:.03%} ({m}) to {max:.1%} ({M})".format(
        min=a.min(),mean=a.mean(),max=a.max(),m=a.argmin(),M=a.argmax()))


    print("At the {region_name} scale, we estimate asset losses due to natural disasters to be {k:.0f} bn$ per year. "\
          "But due to lack of socio economic capacity, welfare losses are {r:.1f} times larger, at {w:.0f} bn$ per year".format( 
    w= results.dWtot_currency.sum()*1e-9,
    k= results.dKtot.sum()*1e-9,
    region_name = region_name,
    r=results.dWtot_currency.sum()/results.dKtot.sum()))

    
print_stats(results)    

for g in country_per_gp.index.unique():
    print("\n=======\n"+g)
    print_stats(results.ix[country_per_gp.ix[g]],g)

for r in country_per_rg.index.unique():
    print("\n=======\n"+r)
    print_stats(results.ix[country_per_rg.ix[r]],r)
    

eca_clients = groups[(~ groups["Income group"].isin(["High income: nonOECD", "High income: OECD"])) 
                         & (groups["Region"]=="Europe & Central Asia")].index
print("\n=======\n"+"ECA")    
print_stats(results.ix[eca_clients],"ECA CLIENTS")    

[c for c in macro.dropna().index if c not in results.dropna().index]

results.query("risk_to_assets>.02").risk_to_assets

a=results.copy()

a["v"] = pd.read_csv("orig_inputs/v_pr_fromPAGER_shaved_GAR.csv", index_col="country").v


to_output = ['gdp_pc_pp', 'pop',"v","fa","resilience","risk","risk_to_assets"]
a=a[to_output]

a["pop"] = a["pop"]/1e6

a["fa_in_gdp"] =  results["fa"]/ results["avg_prod_k"]

a.to_csv("orig_intermediate/main_results.csv")


a.loc[:,["fa","fa_in_gdp","v","resilience","risk", "risk_to_assets"]]=100*a[["fa","fa_in_gdp","v","resilience","risk","risk_to_assets"]]
desc=pd.read_csv("orig_inputs/inputs_info.csv").set_index('key')["descriptor"]
a=a.rename(columns=desc).dropna()
a.to_excel("orig_intermediate/results.xlsx")
a.head()

