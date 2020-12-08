import numpy as np
import pandas as pd

#help with multiindex dataframe
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories

from scipy.interpolate import interp1d

pd.set_option('display.width', 200)

#name of admin division 
economy = "country"
#levels of index at which one event happens
event_level = [economy, "hazard", "rp"]

#return period to use when no rp is provided (mind that this works with protection)
default_rp = "default_rp" 

#categories of households
income_cats   = pd.Index(["poor","nonpoor"],name="income_cat")
#categories for social protection
affected_cats = pd.Index(["a", "na"]            ,name="affected_cat")
helped_cats   = pd.Index(["helped","not_helped"],name="helped_cat")
#infrastructure stocks categories
infra_cats = ['transport','all_infra_but_transport']
sector_cats = ['transport','all_infra_but_transport','other_k']


def compute_resilience(df_in,cat_info, infra_stocks, hazard_ratios=None, is_local_welfare=True, return_iah=False, return_stats=False,optionT="data", optionPDS="no", optionB = "data", loss_measure = "dk",fraction_inside=1, verbose_replace=False, optionFee="tax",  share_insured=.25):
    """Main function. Computes all outputs (dK, resilience, dC, etc,.) from inputs
    optionT=="perfect","data","x33","incl" or "excl"
    optionPDS=="no","unif_all","unif_poor","prop"
    optionB=="data","unif_poor"
    optionFee == "tax" (default) or "insurance_premium"
    fraction_inside=0..1 (how much aid is paid domestically)
    """
    print(optionB)

    #make sure to copy inputs
    macro        =    df_in.dropna().copy(deep=True)
    cat_info     = cat_info.dropna().copy(deep=True)
    try: infra_stocks = infra_stocks.dropna().copy(deep=True)
    except: pass
    
    ####DEFAULT VALUES
    if type(hazard_ratios)==pd.DataFrame:

        #make sure to copy inputs
        hazard_ratios = hazard_ratios.dropna().copy(deep=True)
        
        #other way of passing dummy hazard ratios
        if hazard_ratios.empty:
            hazard_ratios=None
    
    #default hazard
    if hazard_ratios is None:
        hazard_ratios = pd.Series(1,index=pd.MultiIndex.from_product([macro.index,["default_hazard"]],names=[economy, "hazard"]))
                   
    #if fa ratios were provided with no hazard data, they are broadcasted to default hazard
    if "hazard" not in get_list_of_index_names(hazard_ratios):
        hazard_ratios = broadcast_simple(hazard_ratios, pd.Index(["default_hazard"], name="hazard"))  
    
    #if hazard data has no rp, it is broadcasted to default hazard
    if "rp" not in get_list_of_index_names(hazard_ratios):
        hazard_ratios_event = broadcast_simple(hazard_ratios, pd.Index([default_rp], name="rp"))
    else:
        #interpolates data to a more granular grid for return periods that includes all protection values
        hazard_ratios_event = interpolate_rps(hazard_ratios,macro.protection)  #XXX: could move this after dkdw into average over rp (but parallel computing within pandas probably means no difference)
    

    #########
    ## PRE PROCESS and harmonize input values
    #removes countries in macro not in cat_info, otherwise it crashes
    try: common_places = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index and c in infra_stocks.index]
    except: common_places = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index]
    macro = macro.ix[common_places]        
    cat_info = cat_info.ix[common_places]        
    hazard_ratios = hazard_ratios.ix[common_places]
  
    try: infra_stocks = infra_stocks.ix[common_places]        
    except: pass
    
    ##consistency of income, gdp, etc.
    # gdp from k and mu
    macro["gdp_pc_pp"]= macro["avg_prod_k"]*agg_to_economy_level(cat_info,"k")

    # conso from k and macro
    cat_info["c"]=(1-macro["tau_tax"])*macro["avg_prod_k"]*cat_info["k"]+ cat_info["gamma_SP"]*macro["tau_tax"]*macro["avg_prod_k"]*agg_to_economy_level(cat_info,"k")  

    #add finance to diversification and taxation
    cat_info["social"] = unpack_social(macro,cat_info)
    cat_info["social"]+= 0.1* cat_info["axfin"]
    macro["tau_tax"], cat_info["gamma_SP"] = social_to_tx_and_gsp(cat_info)

    #RECompute consumption from k and new gamma_SP and tau_tax
    cat_info["c"]=(1-macro["tau_tax"])*macro["avg_prod_k"]*cat_info["k"]+ cat_info["gamma_SP"]*macro["tau_tax"]*macro["avg_prod_k"]*agg_to_economy_level(cat_info,"k")  

    # # # # # # # # # # # # # # # # # # #
    # MACRO_MULTIPLIER
    # # # # # # # # # # # # # # # # # # #
    
    #rebuilding exponentially to 95% of initial stock in reconst_duration
    three = np.log(1/0.05) 
    recons_rate = three/ macro["T_rebuild_K"]  
    
    # Calculation of macroeconomic resilience
    try:
        macro["v_product"]        = v_product(infra_stocks, infra_cats)
        macro["alpha_v_sum"]      = alpha_v_sum(infra_stocks)
        macro["dy_over_dk"]       = (1-macro["v_product"])/macro["alpha_v_sum"]*macro["avg_prod_k"]+macro["v_product"]*macro["avg_prod_k"]/3
    # macro["dy_over_dk"]       = macro["avg_prod_k"]
        macro["macro_multiplier"] = (macro["dy_over_dk"] +recons_rate)/(macro["rho"]+recons_rate)  
    except: macro["macro_multiplier"] = (macro["avg_prod_k"] +recons_rate)/(macro["rho"]+recons_rate)  

    ####FORMATING
    #gets the event level index
    event_level_index = hazard_ratios_event.reset_index().set_index(event_level).index
    
    #Broadcast macro to event level 
    macro_event = broadcast_simple(macro,  event_level_index)
    #updates columns in macro with columns in hazard_ratios_event
    cols = [c for c in macro_event if c in hazard_ratios_event]
    if not cols==[]:
        macro_event[cols] =  hazard_ratios_event[cols]
    if verbose_replace:
        print("Replaced in macro: "+", ".join(cols))
    
    #Broadcast categories to event level
    cats_event = broadcast_simple(cat_info,  event_level_index)
    # applies mh ratios to relevant columns
    cols_c = [c for c in cats_event if c in hazard_ratios_event] #columns that are both in cats_event and hazard_ratios_event
    
    
    if not cols_c==[]:
        hrb = broadcast_simple( hazard_ratios_event[cols_c], cat_info.index).reset_index().set_index(get_list_of_index_names(cats_event)) #explicitly broadcasts hazard ratios to contain income categories
        cats_event[cols_c] = hrb
        if verbose_replace:
            print("Replaced in cats: "+", ".join(cols_c))
    if verbose_replace:
        print("Replaced in both: "+", ".join(np.intersect1d(cols,cols_c)))
           
  
    ####COMPUTING LOSSES
    #computes dk and dW per event
    out=compute_dK_dW(macro_event, cats_event, optionT=optionT, optionPDS=optionPDS, optionB=optionB, return_iah=return_iah,  return_stats= return_stats,is_local_welfare=is_local_welfare, loss_measure=loss_measure,fraction_inside=fraction_inside, optionFee=optionFee,  share_insured=share_insured)
    ###TODO: split this function. First compute dK. Then compute response. Then compute DW. 3 functions. (then aggregate over rp and hazard)
    
    #unpacks if needed
    if return_iah:
        dkdw_event,cats_event_iah  = out
    else:
        dkdw_event = out
    
    ##AGGREGATES LOSSES
    #Averages over return periods to get dk_{hazard} and dW_{hazard}
    dkdw_h = average_over_rp(dkdw_event,macro_event["protection"])
    
    #Sums over hazard dk, dW (gets one line per economy)
    #remove inputs from this variable to avoid summing inputs over economy
    #to do it by sector, add an agregation function that multiplies by shares and then sums over sectors
    dkdw = dkdw_h.sum(level=economy)

    #adds dk and dw-like columns to macro
    macro[dkdw.columns]=dkdw
    
    #computes socio economic capacity and risk at economy level
    results = calc_risk_and_resilience_from_k_w(macro, is_local_welfare)
    
    ###OUTPUTS
    if return_iah:
        return results, cats_event_iah
    else:
        return results

    
def compute_dK_dW(macro_event, cats_event, optionT="data", optionPDS='no', optionB="data", optionFee="tax", return_iah=False, return_stats=False, is_local_welfare=True,loss_measure="dk",fraction_inside=1, share_insured=.25):  
    '''Computes dk and dW line by line. 
    presence of multiple return period or multihazard data is transparent to this function'''    
    ###TODO: split this function. First compute dK. Then compute response. Then compute DW. 3 functions. (then aggregate over rp and hazard)

    ################## MICRO
    ####################
    #### Consumption losses per AFFECTED CATEGORIES before response
    cats_event_ia=concat_categories(cats_event,cats_event, index= affected_cats)
    #counts affected and non affected
    naf = cats_event["n"]*cats_event.fa
    nna = cats_event["n"]*(1-cats_event.fa)
    cats_event_ia["n"] = concat_categories(naf,nna, index= affected_cats)
    
    #de_index so can access cats as columns and index is still event
    cats_event_ia = cats_event_ia.reset_index(["income_cat", "affected_cat"]).sort_index()
    
    #post early-warning vulnerability 
    cats_event_ia["v_shew"]=cats_event_ia["v"]*(1-macro_event["pi"]*cats_event_ia["shew"])
    
    # print(a)
    
    #capital losses and total capital losses (mind correcting unaffected dk to zero)
    cats_event_ia["dk"]  = cats_event_ia[["k","v_shew"]].prod(axis=1, skipna=False)
    #sets unaffected dk to 0
    cats_event_ia.ix[(cats_event_ia.affected_cat=='na') ,"dk" ]=0

    #"national" losses (to scale down transfers)
    macro_event["dk_event"] =  agg_to_event_level(cats_event_ia, "dk")
    
    #immediate consumption losses: direct capital losses plus losses through event-scale depression of transfers
    cats_event_ia["dc"] = (1-macro_event["tau_tax"])*cats_event_ia["dk"]  +  cats_event_ia["gamma_SP"]*macro_event["tau_tax"] *macro_event["dk_event"] 
    

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    cats_event_ia["dc_npv_pre"] = cats_event_ia["dc"]*macro_event["macro_multiplier"]


    #POST DISASTER RESPONSE

    
    #adding hELPED/NOT HELPED CATEGORIES, indexed at event level 
    # !!!!!!!MIND THAT N IS 2 AT THIS LEVEL !!!!!!!!!!!!!!
    cats_event_iah = concat_categories(cats_event_ia,cats_event_ia, index= helped_cats).reset_index(helped_cats.name).sort_index()
    cats_event_iah["help_needed"] = 0
    cats_event_iah["help_received"] = 0
    cats_event_iah["help_fee"] =0
    #baseline case (no insurance)
    if optionFee!="insurance_premium":
        macro_event, cats_event_iah = compute_response(macro_event, cats_event_iah,  optionT=optionT, optionPDS=optionPDS, optionB=optionB, optionFee=optionFee, fraction_inside=fraction_inside, loss_measure = loss_measure)
        
    #special case of insurance that adds to existing default PDS
    else:
        #compute post disaster response with default PDS from data ONLY
        m__,c__ = compute_response(macro_event, cats_event_iah,optionT="data", optionPDS=optionPDS, optionB="data", optionFee="tax", fraction_inside=1, loss_measure="dk") 

        c__h = c__.rename(columns=dict(helped_cat="has_received_help_from_PDS_cat"))
        
        cats_event_iah_h = concat_categories(c__h,c__h, index= helped_cats).reset_index(helped_cats.name).sort_index()

        #compute post disaster response with insurance ONLY
        macro_event, cats_event_iah = compute_response(
            macro_event.assign(shareable=share_insured),cats_event_iah_h ,
            optionT=optionT, optionPDS=optionPDS, optionB=optionB, optionFee=optionFee, fraction_inside=fraction_inside, loss_measure = loss_measure)
        
        columns_to_add = ["need","aid"]
        macro_event[columns_to_add] +=  m__[columns_to_add]
        
        # columns_to_add_iah = ["help_received","help_fee"]
        # cats_event_iah[columns_to_add_iah] += c__[columns_to_add_iah]
    
    
       
    #effect on welfare
    cats_event_iah["dc_npv_post"] = cats_event_iah["dc_npv_pre"] -  cats_event_iah["help_received"]  + cats_event_iah["help_fee"]
    
    
    # print(cats_event_iah.head())
    # print("\n macro \n")
    # print(macro_event.head())
    
    
    cats_event_iah["dw"] = calc_delta_welfare(cats_event_iah, macro_event) 
    
    #aggregates dK and delta_W at df level
    dK      = agg_to_event_level(cats_event_iah,"dk")
    delta_W = agg_to_event_level(cats_event_iah,"dw")

    
    ###########
    #OUTPUT
    df_out = pd.DataFrame(index=macro_event.index)
    
    df_out["dK"] = dK
    df_out["dKtot"]=dK*macro_event["pop"] #/macro_event["protection"]

    df_out["delta_W"]    =delta_W
    df_out["delta_W_tot"]=delta_W*macro_event["pop"] #/macro_event["protection"]

    df_out["average_aid_cost_pc"] = macro_event["aid"]
    
    if return_stats:
        stats = np.setdiff1d(cats_event_iah.columns,event_level+['helped_cat',  'affected_cat',     'income_cat'])
        df_stats = agg_to_event_level(cats_event_iah, stats)
        # if verbose_replace:
        print("!! (maybe broken) stats are "+",".join(stats))
        df_out[df_stats.columns]=(df_stats.T*macro_event.protection).T #corrects stats from protecgion because they get averaged over rp with the rest of df_out later
    
    if return_iah:
        return df_out,cats_event_iah
    else: 
        return df_out
    

def compute_response(macro_event, cats_event_iah,  optionT="data", optionPDS='no', optionB="data", optionFee="tax", fraction_inside=1, loss_measure="dk"):    
    """
    Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing, and dimensioning of the help.
    Returns copies of macro_event and cats_event_iah updated with stuff
    TODO In general this function is ill coded and should be rewritteN
    """       
    print(optionB)
    
    macro_event    = macro_event.copy()
    # cats_event_ia = cats_event_ia.copy()
    cats_event_iah = cats_event_iah.copy()
    
    macro_event["fa"] =  agg_to_event_level(cats_event_iah,"fa") /2 # fact of life: _h means 2
    # macro_event["fa"] =  agg_to_event_level(cats_event_ia,"fa") 
    
        
    ####targeting errors
    if optionT=="perfect":
        macro_event["error_incl"] = 0
        macro_event["error_excl"] = 0    
    elif optionT=="prop_nonpoor_lms":
        macro_event["error_incl"] = 0
        macro_event["error_excl"] = 1-25/80  #25% of pop chosen among top 80 DO receive the aid
    elif optionT=="data":
        macro_event["error_incl"]=(1-macro_event["prepare_scaleup"])/2*macro_event["fa"]/(1-macro_event["fa"])
        macro_event["error_excl"]=(1-macro_event["prepare_scaleup"])/2
    elif optionT=="x33":
        macro_event["error_incl"]= .33*macro_event["fa"]/(1-macro_event["fa"])
        macro_event["error_excl"]= .33
    elif optionT=="incl":
        macro_event["error_incl"]= .33*macro_event["fa"]/(1-macro_event["fa"])
        macro_event["error_excl"]= 0
    elif optionT=="excl":
        macro_event["error_incl"]= 0
        macro_event["error_excl"]= 0.33
    else:
        print("unrecognized targeting error option "+optionT)
        return None
    
    #counting (mind self multiplication of n)
    cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='a') ,"n"]*=(1-macro_event["error_excl"])
    cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='a') ,"n"]*=(  macro_event["error_excl"])
    cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='na'),"n"]*=(  macro_event["error_incl"])  
    cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='na'),"n"]*=(1-macro_event["error_incl"])
    ###!!!! n is one again from here.

    # #should be only ones
    # cats_event_iah.n.sum(level=event_level)
    
    # MAXIMUM NATIONAL SPENDING ON SCALE UP
    macro_event["max_aid"] = macro_event["max_increased_spending"]*macro_event["borrow_abi"]*macro_event["gdp_pc_pp"]

    ##THIS LOOP DETERMINES help_received and help_fee by category   (currently may also output cats_event_ia[["need","aid","unif_aid"]] which might not be necessary ) 
    # how much post-disaster support?
    
    if optionB=="unif_poor":
        ### CALCULATE FIRST THE BUDGET FOR unif_poor and use the same budget for other methods
        d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a') & (cats_event_iah.income_cat=='poor')]        
        macro_event["need"] = macro_event["shareable"]*agg_to_event_level(d,loss_measure)
        macro_event["aid"] = macro_event["need"].clip(upper=macro_event["max_aid"]) 
    elif optionB=="one_per_affected":
        ### CALCULATE FIRST THE BUDGET FOR unif_poor and use the same budget for other methods
        d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a')]        
        d["un"]=1
        macro_event["need"] = agg_to_event_level(d,"un")
        macro_event["aid"] = macro_event["need"] 
    elif optionB=="one":
        macro_event["aid"] = 1
    elif optionB=="x10":
        macro_event["aid"] = 0.1*macro_event["gdp_pc_pp"]
    elif optionB=="x05":
        macro_event["aid"] = 0.05*macro_event["gdp_pc_pp"]
    elif optionB=="max01":
        macro_event["max_aid"] = 0.01*macro_event["gdp_pc_pp"]
    elif optionB=="max05":
        macro_event["max_aid"]=0.05*macro_event["gdp_pc_pp"]
    elif optionB=="unlimited":
        d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a')]
        macro_event["need"] = macro_event["shareable"]*agg_to_event_level(d,loss_measure)
        macro_event["aid"] = macro_event["need"]         
        
    if optionFee == "tax":
        pass
    elif optionFee == "insurance_premium":
        pass
    
    if optionPDS=="no":        
        macro_event["aid"] = 0
        
    elif optionPDS in ["unif_all", "unif_poor"]:
        
        if optionPDS=="unif_all":
            #individual need: NPV losses for affected
            d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a')]
        elif optionPDS=="unif_poor":
            #NPV losses for POOR affected
            d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a') & (cats_event_iah.income_cat=='poor')]
            # ^ THIS IS THE CULPRIT LINE! It's going to calculate need based only on poor hh, but rich hh are also getting a payout in unif_poor
            # The pot of money will be too small when it gets distributed to rich and poor hh later

        #aggs need of those selected in the previous block (eg only poor) at event level 
        macro_event["need"] = macro_event["shareable"]*agg_to_event_level(d,loss_measure)

        #actual aid reduced by capacity
        if optionB=="data":
            macro_event["aid"] = (macro_event["need"]*macro_event["prepare_scaleup"]*macro_event["borrow_abi"]).clip(upper=macro_event["max_aid"])             
        elif optionB in ["max01" , "max05"]:
            macro_event["aid"] = (macro_event["need"]).clip(upper=macro_event["max_aid"]) 
            # otherwise we keep the aid from the unif_poor calculation (or one)

        #aid divided by people aided
        macro_event["unif_aid"] = macro_event["aid"]/(cats_event_iah.ix[cats_event_iah.helped_cat=="helped","n"].sum(level=event_level))
        
        #help_received: all who receive receive same
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped'),"help_received"]+= macro_event["unif_aid"]

        #aid funding
        cats_event_iah["help_fee"] += fraction_inside*macro_event["aid"]*cats_event_iah["k"]/agg_to_event_level(cats_event_iah,"k")
        
    # $1 per helped person
    elif optionPDS=="one":  
        macro_event["unif_aid"] = 1
        #help_received: all who receive receive same
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped'),"help_received"]+= macro_event["unif_aid"]
        macro_event["need"] = agg_to_event_level(cats_event_iah,"help_received")
        macro_event["aid"] = macro_event["need"] 
        cats_event_iah["help_fee"]+= fraction_inside*macro_event["aid"]*cats_event_iah["k"]/agg_to_event_level(cats_event_iah,"k")

    elif optionPDS=="hundred":  
        macro_event["unif_aid"] = macro_event["gdp_pc_pp"]
        #help_received: all who receive receive same
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped'),"help_received"]+= macro_event["unif_aid"]
        macro_event["need"] = agg_to_event_level(cats_event_iah,"help_received")
        macro_event["aid"] = macro_event["need"] 
        cats_event_iah["help_fee"]+= fraction_inside*macro_event["aid"]*cats_event_iah["k"]/agg_to_event_level(cats_event_iah,"k")
                
    elif optionPDS in ["prop","perfect", "prop_nonpoor", "prop_nonpoor_lms"]:

        #needs based on losses per income category (needs>0 for non affected people)
        if optionPDS in ["prop_nonpoor","prop_nonpoor_lms" ]: 
            cats_event_iah.ix[(cats_event_iah.income_cat=='poor'), "need"]   =0 
        else :
            cats_event_iah.ix[(cats_event_iah.income_cat=='poor')& (cats_event_iah.affected_cat=='a'), "need"]   =   cats_event_iah.ix[(cats_event_iah.income_cat=='poor')   & (cats_event_iah.affected_cat=='a') ,loss_measure]#.sum(level=event_level)
            cats_event_iah.ix[(cats_event_iah.income_cat=='poor')& (cats_event_iah.affected_cat=='na'),"need"]   =   cats_event_iah.ix[(cats_event_iah.income_cat=='poor')   & (cats_event_iah.affected_cat=='a') ,loss_measure]#.sum(level=event_level)
        
        cats_event_iah.ix[(cats_event_iah.income_cat=='nonpoor')& (cats_event_iah.affected_cat=='a'),"need"]  =cats_event_iah.ix[(cats_event_iah.income_cat=='nonpoor')& (cats_event_iah.affected_cat=='a') ,loss_measure]#.sum(level=event_level)
        cats_event_iah.ix[(cats_event_iah.income_cat=='nonpoor')& (cats_event_iah.affected_cat=='na'),"need"] =cats_event_iah.ix[(cats_event_iah.income_cat=='nonpoor')& (cats_event_iah.affected_cat=='a') ,loss_measure]#.sum(level=event_level)
        
        
        d = cats_event_iah.ix[cats_event_iah.helped_cat=="helped",["need","n"]]
            # "national" needs: agg over helped people
        macro_event["need"] = macro_event["shareable"]*agg_to_event_level(d,"need")
       
        # actual aid is national need reduced by capacity
        if optionB=="data":
            macro_event["aid"] = (macro_event["need"]*macro_event["prepare_scaleup"]*macro_event["borrow_abi"]).clip(upper=macro_event["max_aid"]) 
        elif optionB in ["max01" , "max05"]:
            macro_event["aid"] = (macro_event["need"]).clip(upper=macro_event["max_aid"]) 
        elif optionB == "unlimited":
            macro_event["aid"]=macro_event["need"]
    
        #actual individual aid reduced prorate by capacity (mind fixing to zero when not helped)
        copy_for_new_help = cats_event_iah.copy(deep=True)
        copy_for_new_help["help_received"] = macro_event["shareable"]*cats_event_iah["need"]*  (macro_event["aid"]/macro_event["need"])  #individual (line in cats_event_iah) need scaled by "national" (cats_event_ia line) 
        where= (cats_event_iah.helped_cat=='not_helped')
        copy_for_new_help.ix[where,"help_received"]=0
        
        cats_event_iah["help_received"]+=copy_for_new_help["help_received"]
        # financed at prorata of individual assets over "national" assets

        if optionFee=="tax":
            cats_event_iah["help_fee"] += fraction_inside * agg_to_event_level(copy_for_new_help,"help_received")*cats_event_iah["k"]/agg_to_event_level(cats_event_iah,"k")
        
        elif optionFee=="insurance_premium":
            
            cats_event_iah.ix[(cats_event_iah.income_cat=='poor'),"help_fee"] += fraction_inside * agg_to_event_level(copy_for_new_help.query("income_cat=='poor'"),"help_received")
            
            cats_event_iah.ix[(cats_event_iah.income_cat=='nonpoor'),"help_fee"]+= fraction_inside * agg_to_event_level(copy_for_new_help.query("income_cat=='nonpoor'"),"help_received")
            
        else:
            print("did not know how to finance the PDS")
            
    else:
        print("unrecognised optionPDS treated as no")
             
    macro_event.to_csv('/Users/brian/Desktop/Dropbox/Bank/resilience_model/results/macro.csv')
    cats_event_iah.to_csv('/Users/brian/Desktop/Dropbox/Bank/resilience_model/results/cats_event_iah.csv')
        
    return macro_event, cats_event_iah
    


def calc_risk_and_resilience_from_k_w(df, is_local_welfare): 
    """Computes risk and resilience from dk, dw and protection. Line by line: multiple return periods or hazard is transparent to this function"""
    
    df=df.copy()
    
    ############################
    #Expressing welfare losses in currency 
    
    #discount rate
    rho = df["rho"]
    h=1e-4
    
    #Reference losses
    h=1e-4
    
    if is_local_welfare:
        wprime =(welf(df["gdp_pc_pp"]/rho+h,df["income_elast"])-welf(df["gdp_pc_pp"]/rho-h,df["income_elast"]))/(2*h)
        # wprime =(welf(df["gdp_pc_pp_ref"]/rho+h,df["income_elast"])-welf(df["gdp_pc_pp_ref"]/rho-h,df["income_elast"]))/(2*h)
    else:
        wprime =(welf(df["gdp_pc_pp_nat"]/rho+h,df["income_elast"])-welf(df["gdp_pc_pp_nat"]/rho-h,df["income_elast"]))/(2*h)
        
    dWref   = wprime*df["dK"]
    
    #expected welfare loss (per family and total)
    df["dWpc_currency"] = df["delta_W"]/wprime  #//df["protection"]
    df["dWtot_currency"]=df["dWpc_currency"]*df["pop"];

    #welfare loss (per family and total)
    #df["dWpc_currency"] = df["delta_W"]/wprime/df["protection"]
    #df["dWtot_currency"]=df["dWpc_currency"]*df["pop"];
    
    #Risk to welfare as percentage of local GDP
    df["risk"]= df["dWpc_currency"]/(df["gdp_pc_pp"]);
    
    ############
    #SOCIO-ECONOMIC CAPACITY)
    df["resilience"] =dWref/(df["delta_W"] );

    ############
    #RISK TO ASSETS
    df["risk_to_assets"]  =df.resilience* df.risk;
        
    return df
    
    

def calc_delta_welfare(micro, macro):
    """welfare cost from consumption before (c) 
    an after (dc_npv_post) event. Line by line"""
  
    

         
    #computes welfare losses per category        
    dw = welf(micro["c"]                       /macro["rho"], macro["income_elast"]) -\
         welf(micro["c"]/macro["rho"]-(micro["dc_npv_post"]), macro["income_elast"])
         
    return dw
    
    
def welf(c,elast):
    """"Welfare function"""
    
    y=(c**(1-elast)-1)/(1-elast)
    
    #log welfare func
    # cond = (elast==1)
    # y[cond] = np.log(c[cond]) 
    
    return y
    
def v_product(infra_stocks, infra_cats):
    """multiplier of the production function, using the vulnerabilities and exposure of infrastructure stocks."""
    p = (infra_stocks.v*infra_stocks.fa).unstack("sector")
    e = infra_stocks.e.unstack("sector")
    q = 1
    for i in infra_cats:
        q = q*(1-p[i])**(e[i])
    return q
    
def alpha_v_sum(infra_stocks):
    """sum of the shares times vulnerabilities times exposure. enters the deltaY over delta K function"""
    a = infra_stocks.drop('e',axis=1).prod(axis=1).sum(level="country")
    return a

    
def agg_to_event_level (df, seriesname):
    """ aggregates seriesname in df (string of list of string) to event level (country, hazard, rp) using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df["n"]).T.sum(level=event_level)
    
def agg_to_economy_level (df, seriesname):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df["n"]).T.sum(level=economy)
    
    
def interpolate_rps(fa_ratios,protection_list):
    
    ###INPUT CHECKING
    if fa_ratios is None:
        return None
    
    if default_rp in fa_ratios.index:
        return fa_ratios
    
    flag_stack= False
    if "rp" in get_list_of_index_names(fa_ratios):
        fa_ratios = fa_ratios.unstack("rp")
        flag_stack = True
 
    if type(protection_list) in [pd.Series, pd.DataFrame]:
        protection_list=protection_list.squeeze().unique().tolist()
        
    #in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(fa_ratios.columns)==pd.MultiIndex:
        keys = fa_ratios.columns.get_level_values(0).unique()
        return pd.concat({col:interpolate_rps(fa_ratios[col],protection_list) for col in  keys}, axis=1).stack("rp")


    ### ACTAL FUNCTION    
    #figures out all the return periods to be included
    all_rps = list(set(protection_list+fa_ratios.columns.tolist()))
    
    fa_ratios_rps = fa_ratios.copy()
    
    #extrapolates linear towards the 0 return period exposure  (this creates negative exposure that is tackled after interp) (mind the 0 rp when computing probas)
    if len(fa_ratios_rps.columns)==1:
        fa_ratios_rps[0] = fa_ratios_rps.squeeze()
    else:
        fa_ratios_rps[0]=fa_ratios_rps.iloc[:,0]- fa_ratios_rps.columns[0]*(
        fa_ratios_rps.iloc[:,1]-fa_ratios_rps.iloc[:,0])/(
        fa_ratios_rps.columns[1]-fa_ratios_rps.columns[0])
    
    
    #add new, interpolated values for fa_ratios, assuming constant exposure on the right
    x = fa_ratios_rps.columns.values
    y = fa_ratios_rps.values
    fa_ratios_rps= pd.concat(
        [pd.DataFrame(interp1d(x,y,bounds_error=False)(all_rps),index=fa_ratios_rps.index, columns=all_rps)]
        ,axis=1).sort_index(axis=1).clip(lower=0).fillna(method="pad",axis=1)
    fa_ratios_rps.columns.name="rp"

    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack("rp")
    
    return fa_ratios_rps    
    from scipy.integrate import simps
    
        
def average_over_rp(df,protection=None):        
    """Aggregation of the outputs over return periods"""
    
    if protection is None:
        protection=pd.Series(0,index=df.index)    
    
    #does nothing if df does not contain data on return periods
    try:
        if "rp" not in df.index.names:
            print("rp was not in df")
            return df
    except(TypeError):
        pass
    
    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values("rp"):
        print("default_rp detected, droping rp")
        return (df.T/protection).T.reset_index("rp",drop=True)
        
    
    df=df.copy().reset_index("rp")
    protection=protection.copy().reset_index("rp",drop=True)
    
    #computes probability of each return period
    return_periods=np.unique(df["rp"].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps

    #matches return periods and their probability
    proba_serie=df["rp"].replace(proba)

    #removes events below the protection level
    proba_serie[protection>df.rp] =0

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels==[0]:
        idxlevels =0
        
    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0).sum(level=idxlevels) # obsolete .div(proba_serie.sum(level=idxlevels),axis=0)
    
    return averaged.drop("rp",axis=1)


def unpack_social(m,cat):
        """Compute social from gamma_SP, taux tax and k and avg_prod_k
        """
        #############
        #### preparation

        #current conso and share of average transfer
        c  = cat.c
        gs = cat.gamma_SP
        
        #social_p
        social = gs* m.gdp_pc_pp  *m.tau_tax /c

        return social
        
def social_to_tx_and_gsp(cat_info):       
        """(tx_tax, gamma_SP) from cat_info[["social","c","n"]] """
        
        tx_tax = cat_info[["social","c","n"]].prod(axis=1, skipna=False).sum(level=economy) / \
                 cat_info[         ["c","n"]].prod(axis=1, skipna=False).sum(level=economy)

        #income from social protection PER PERSON as fraction of PER CAPITA social protection
        gsp=     cat_info[["social","c"]].prod(axis=1,skipna=False) /\
             cat_info[["social","c","n"]].prod(axis=1, skipna=False).sum(level=economy)
        
        return tx_tax, gsp

        
 
def unpack(v,pv,fa,pe,ph,share1):
#returns v_p,v_r, far, fap, cp, cr from the inputs
# v_p,v_r, far, fap, cp, cr = unpack(v,pv,fa,pe,ph,share1)
    
    v_p = v*(1+pv)
    
    fap_ref= fa*(1+pe)
    
    
    far_ref=(fa-ph*fap_ref)/(1-ph)
    cp_ref=   share1 /ph
    cr_ref=(1-share1)/(1-ph)
    
    x=ph*cp_ref 
    y=(1-ph)*cr_ref  
    
    v_r = ((x+y)*v - x* v_p)/y
    
    return v_p,v_r, fap_ref, far_ref, cp_ref, cr_ref

def compute_v_fa(df):
    
    fap = df["fap"]
    far = df["far"]
    
    vp = df.v_p
    vr=df.v_r

    ph = 0.2#df["pov_head"]
        
    cp=    df["gdp_pc_pp"]*df["share1"]/ph 
    cr= df["gdp_pc_pp"]*(1-df["share1"])/(1-ph)
    
    fa = ph*fap+(1-ph)*far
    
    x=ph*cp 
    y=(1-ph)*cr 
    
    v=(y*vr+x*vp)/(x+y)
    
    pv = vp/v-1
    pe = fap/fa-1
    
    
    return v,pv,fa,pe
     
     
     
def compute_resilience_from_packed_inputs(df,test_input=False) :        


    df=df.dropna().copy()

    ##MACRO
    macro_cols = [c for c in df if "macro" in c ]
    macro = df[macro_cols]
    macro = macro.rename(columns=lambda c:c.replace("macro_",""))

    ##CAT INFO
    cat_cols = [c for c in df if "cat_info" in c ]
    cat_info = df[cat_cols]
    cat_info.columns=pd.MultiIndex.from_tuples([c.replace("cat_info_","").split("__") for c in cat_info])
    cat_info = cat_info.sort_index(axis=1).stack()
    cat_info.index.names="country","income_cat"


    ##HAZARD RATIOS

    ###exposure
    fa_cols =  [c for c in df if "hazard_ratio_fa" in c ]
    fa = df[fa_cols]
    fa.columns=[c.replace("hazard_ratio_fa__","") for c in fa]


    ##### add poor and nonpoor
    hop=pd.DataFrame(2*[fa.unstack()], index=["poor","nonpoor"]).T
    hop.ix["flood"]["poor"] = df.hazard_ratio_flood_poor
    # print(hop)
    hop.ix["surge"]["poor"] = hop.ix["flood"]["poor"] * df["ratio_surge_flood"]
    hop.ix["surge"]["nonpoor"] = hop.ix["flood"]["nonpoor"] * df["ratio_surge_flood"]
    hop=hop.stack().swaplevel(0,1).sort_index()
    hop.index.names=["country","hazard","income_cat"]

    hazard_ratios = pd.DataFrame()
    hazard_ratios["fa"]=hop

    ## Shew
    hazard_ratios["shew"]=0
    hazard_ratios["shew"]+=df.shew_for_hazard_ratio
    #no EW for earthquake
    hazard_ratios["shew"]=hazard_ratios.shew.unstack("hazard").assign(earthquake=0).stack("hazard").reset_index().set_index(["country", "hazard","income_cat"])
    
    if test_input:
        return macro, cat_info, hazard_ratios,
        
    else:    

        #ACTUALLY DO THE THING
        out = compute_resilience(macro, cat_info, hazard_ratios)
        
        df[["risk","resilience","risk_to_assets"]] = out[["risk","resilience","risk_to_assets"]]
        
        return df



import itertools
from progress_reporter import *    
def compute_derivative(df_original,score_card_set,deriv_set ,**kwargs):
        
    # kwargs.update(verbose_output=True)
        
    h=2e-3
    
   

    #orginal results
    fx = compute_resilience_from_packed_inputs(df_original,  **kwargs)[score_card_set]

    headr = list(itertools.product(deriv_set,score_card_set))
    #(countries, (input vars, outpus))
    der=  pd.DataFrame(index=df_original.dropna().index, columns=pd.MultiIndex.from_tuples(headr,names=["var","out"])).sortlevel(0,axis=1) 

    for var in deriv_set:
        progress_reporter(var)
        df_h=df_original.copy(deep=True)
        df_h[var]=df_h[var]+h
        fxh= compute_resilience_from_packed_inputs(df_h, **kwargs)[score_card_set]
        der[var] = (fxh-fx)/(h)  #this reads (all countries, (this var, all output))  = (all countries, all output)

    
    der=der.swaplevel("var","out",axis=1).sortlevel(0,axis=1) 

    derivatives =     pd.concat([der], axis=1).sortlevel(0,axis=1)# (countries, (outputs, inputs))

    #Signs of resilience derivative 
    for k in score_card_set:
    
        print("\nRegarding ", k)
        der = np.sign(derivatives[k]).replace(0,np.nan)
        signs= pd.Series(index=der.columns)
        for i in signs.index:
            if (der[i].min()==der[i].max()): #all nonnan signs are equal
                signs[i]=der[i].min()
            else:
                print("ambigous sign for "+i)
                signs[i]=np.nan
            
    return derivatives    
