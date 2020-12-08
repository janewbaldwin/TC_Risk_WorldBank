from res_ind_lib import *
from lib_compute_resilience_and_risk import *

from replace_with_warning import *
import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)
import numpy as np
import pandas as pd

#define directory
use_published_inputs = False

model        = os.getcwd() #get current directory
inputs       = model+'/inputs/' #get inputs data directory
intermediate = model+'/intermediate/' #get outputs data directory

if use_published_inputs:
    inputs       = model+'/orig_inputs/' #get inputs data directory
    intermediate = model+'/orig_intermediate/' #get outputs data directory

#create loop over policies
#for pol_str in ['', '_T_rebuild_K1', '_T_rebuild_K2', '_T_rebuild_K4', '_T_rebuild_K5']: #build back faster
#for pol_str in ['', '_bbb0.2', '_bbb0.4', '_bbb-0.2', '_bbb-0.4']: #build back better

results_policy_summary = pd.DataFrame(index=pd.read_csv(intermediate+"macro.csv", index_col='country').dropna().index)
#for pol_str in ['']:
for pol_str in ['','_bbb_complete1','_bbb_incl1','_bbb_fast1','_bbb_fast2','_bbb_fast4','_bbb_fast5','_bbb_50yrstand1']:

    print(pol_str)
    optionFee="tax"
    optionPDS="unif_poor"

    if optionFee=="insurance_premium":
        optionB='unlimited'
        optionT='perfect'
    else:
        optionB='data'
        optionT='data'

    print('optionFee =',optionFee, 'optionPDS =', optionPDS, 'optionB =', optionB, 'optionT =', optionT)

    #Options and parameters
    economy="country" #province, deparmtent
    event_level = [economy, "hazard", "rp"]	#levels of index at which one event happens
    default_rp = "default_rp" #return period to use when no rp is provided (mind that this works with protection)
    income_cats   = pd.Index(["poor","nonpoor"],name="income_cat")	#categories of households
    affected_cats = pd.Index(["a", "na"]            ,name="affected_cat")	#categories for social protection
    helped_cats   = pd.Index(["helped","not_helped"],name="helped_cat")

    #read data
    macro = pd.read_csv(intermediate+'macro'+pol_str+".csv", index_col=economy).dropna()
    cat_info = pd.read_csv(intermediate+'cat_info'+pol_str+".csv",  index_col=[economy, "income_cat"]).dropna()
    hazard_ratios = pd.read_csv(intermediate+'hazard_ratios'+pol_str+".csv", index_col=event_level+["income_cat"]).dropna()
    groups =  pd.read_csv(inputs+"income_groups.csv",header =4,index_col=2)
    country_per_gp = groups["Income group"].reset_index().dropna().set_index("Income group").squeeze()
    country_per_rg = groups["Region"].reset_index().dropna().set_index("Region").squeeze()

    #compute
    macro_event, cats_event, hazard_ratios_event, macro = process_input(pol_str,macro,cat_info,hazard_ratios,economy,event_level,default_rp,verbose_replace=True) #verbose_replace=True by default, replace common columns in macro_event and cats_event with those in hazard_ratios_event

    macro_event, cats_event_ia = compute_dK(macro_event, cats_event,event_level,affected_cats) #calculate the actual vulnerability, the potential damange to capital, and consumption

    macro_event, cats_event_iah = calculate_response(macro_event,cats_event_ia,event_level,helped_cats,optionFee=optionFee,optionT=optionT, optionPDS=optionPDS, optionB=optionB,loss_measure="dk",fraction_inside=1, share_insured=.25)
    #optionFee: tax or insurance_premium  optionFee="insurance_premium",optionT="perfect", optionPDS="prop", optionB="unlimited",optionFee="tax",optionT="data", optionPDS="unif_poor", optionB="data",
    #optionT(targeting errors):perfect, prop_nonpoor_lms, data, x33, incl, excl.
    #optionB:one_per_affected, one_per_helped, one, unlimited, data, unif_poor, max01, max05
    #optionPDS: unif_poor, no, "prop", "prop_nonpoor"

    macro_event.to_csv('output/macro_'+optionFee+'_'+optionPDS+'_'+pol_str+'.csv',encoding="utf-8", header=True)
    cats_event_iah.to_csv('output/cats_event_iah_'+optionFee+'_'+optionPDS+'_'+pol_str+'.csv',encoding="utf-8", header=True)

    out = compute_dW(macro_event,cats_event_iah,event_level,return_stats=True,return_iah=True)

    #Computes
    args = dict(return_stats=True,hazard_ratios = hazard_ratios)
    #results, iah=compute_resilience(macro,cat_info,None,return_iah=True,verbose_replace=True,**args)
    results,iah = process_output(macro,out,macro_event,economy,default_rp,return_iah=True,is_local_welfare=True)

    #Saves
    results.to_csv('output/results_'+optionFee+'_'+optionPDS+'_'+pol_str+'.csv',encoding="utf-8", header=True)
    iah.to_csv('output/iah_'+optionFee+'_'+optionPDS+'_'+pol_str+'.csv',encoding="utf-8", header=True)

    results_policy_summary[pol_str+'_dw_tot_curr'] = results['dWtot_currency']
results_policy_summary.to_csv('output/results_policy_summary.csv')
    # result1=pd.read_csv("output-old/results.csv", index_col=economy)
    # iah1=pd.read_csv("output-old/iah.csv", index_col=event_level+["income_cat","affected_cat","helped_cat"])
    # print(((result1-results)/results).max())
    # print(((iah1-iah.reset_index().set_index(event_level+["income_cat","affected_cat","helped_cat"]))/iah1).max())
