#This script provides data input for the resilience indicator multihazard model. The script was developed by Adrien Vogt-Schilb and improved by Jinqiang Chen.
#Import package for data analysis
from lib_gather_data import *
from replace_with_warning import *
from apply_policy import *
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)
from lib_gar_preprocess import *

#define directory
use_2016_inputs = False
use_2016_ratings = False
constant_fa =True

year_str = ''
if use_2016_inputs: year_str = 'orig_'

model        = os.getcwd() #get current directory
inputs       = model+'/'+year_str+'inputs/' #get inputs data directory
intermediate = model+'/'+year_str+'intermediate/' #get outputs data directory

if not os.path.exists(intermediate): os.makedirs(intermediate)
# ^ if the depository directory doesn't exist, create one

# Run GAR preprocessing
gar_preprocessing(inputs,intermediate)

debug = False

#Options and parameters
protection_from_flopros=True #FLOPROS is an evolving global database of flood potection standards. It will be used in Protection.
no_protection=True #Used in Protection.
use_GLOFRIS_flood=False  #else uses GAR (True does not work i think)
use_guessed_social=True #else keeps nans
use_avg_pe=True #otherwise 0 when no data
use_newest_wdi_findex_aspire=False  #too late to include new data just before report release
drop_unused_data=True #if true removes from df and cat_info theintermediate variables
economy="country" #province, deparmtent
event_level = [economy, "hazard", "rp"]	#levels of index at which one event happens
default_rp = "default_rp" #return period to use when no rp is provided (mind that this works with protection)
income_cats   = pd.Index(["poor","nonpoor"],name="income_cat")	#categories of households
affected_cats = pd.Index(["a", "na"]            ,name="affected_cat")	#categories for social protection
helped_cats   = pd.Index(["helped","not_helped"],name="helped_cat")

poverty_head=0.2
reconstruction_time=3.0
reduction_vul=0.2
inc_elast=1.5
discount_rate=0.06
asset_loss_covered=0.8
max_support=0.05
fa_threshold =  0.9

#Country dictionaries
any_to_wb=pd.read_csv(inputs+"any_name_to_wb_name.csv",index_col="any") #Names to WB names

for _c in any_to_wb.index:
    __c = _c.replace(' ','')
    if __c != _c:
        try: any_to_wb.loc[__c] = any_to_wb.loc[_c,'wb_name']
        except: pass

any_to_wb = any_to_wb.squeeze()

iso3_to_wb=pd.read_csv(inputs+"iso3_to_wb_name.csv").set_index("iso3").squeeze()	#iso3 to wb country name table
iso2_iso3=pd.read_csv(inputs+"names_to_iso.csv", usecols=["iso2","iso3"]).drop_duplicates().set_index("iso2").squeeze() #iso2 to iso3 table

#Read data
##Macro data
###Economic data from the world bank
the_file=inputs+"wb_data.csv"#"wb_data_backup.csv"
nb_weeks=(time.time()-os.stat(the_file).st_mtime )/(3600*24*7)	#calculate the nb of weeks since the last modified time
if nb_weeks>20:
    warnings.warn("World bank data are "+str(int(nb_weeks))+" weeks old. You may want to download them again.")
df=pd.read_csv(the_file).set_index(economy)
df["urbanization_rate"]=pd.read_csv(inputs+"wb_data.csv").set_index(economy)["urbanization_rate"]
df=df.drop([i for i in ["plgp","unemp","bashs","ophe", "axhealth",'share1_orig'] if i in df.columns],axis=1)	## Drops here the data not used, to avoid it counting as missing data. What are included are:gdp_pc_pp, pop, share1, axfin_p, axfin_r, social_p, social_r, urbanization_rat.

###Define parameters
df["pov_head"]=poverty_head #poverty head
ph=df.pov_head
df["T_rebuild_K"] = reconstruction_time #Reconstruction time
df["pi"] = reduction_vul	# how much early warning reduces vulnerability
df["income_elast"] = inc_elast	#income elasticity
df["rho"] = discount_rate	#discount rate
df["shareable"]=asset_loss_covered  #target of asset losses to be covered by scale up
df["max_increased_spending"] = max_support # 5% of GDP in post-disaster support maximum, if everything is ready

###Social transfer Data from EUsilc (European Union Survey of Income and Living Conditions) and other countries.
silc=pd.read_csv(inputs+"social_ratios.csv") #XXX: there is data from ASPIRE in social_ratios. Use fillna instead to update df.
silc=silc.set_index(silc.cc.replace({"EL":"GR","UK":"GB"}).replace(iso2_iso3).replace(iso3_to_wb)) #Change indexes with wold bank names. UK and greece have differnt codes in Europe than ISO2. The first replace is to change EL to GR, and change UK to GB. The second one is to change iso2 to iso3, and the third one is to change iso3 to the wb
df.ix[silc.index,["social_p","social_r"]]  = silc[["social_p","social_r"]] #Update social transfer from EUsilc.
where=(isnull(df.social_r)&~isnull(df.social_p))|(isnull(df.social_p)&~isnull(df.social_r)) #shows the country where social_p and social_r are not both NaN.
print("social_p and social_r are not both NaN for " + "; ".join(df.loc[where].index))
df.loc[isnull(df.social_r),['social_p','social_r']]=np.nan
df.loc[isnull(df.social_p),['social_p','social_r']]=np.nan

###Guess social transfer
guessed_social=pd.read_csv(inputs+"df_social_transfers_statistics.csv", index_col=0)[["social_p_est","social_r_est"]]
guessed_social.columns=["social_p", "social_r"]
if use_guessed_social:
    df=df.fillna(guessed_social.clip(lower=0, upper=1)) #replace the NaN with guessed social transfer.

####HFA (Hyogo Framework for Action) data to assess the role of early warning system
#2015 hfa
hfa15=pd.read_csv(inputs+"HFA_all_2013_2015.csv")
hfa15=hfa15.set_index(replace_with_warning(hfa15["Country name"],any_to_wb))
# READ THE LAST HFA DATA
hfa_newest=pd.read_csv(inputs+"HFA_all_2011_2013.csv")
hfa_newest=hfa_newest.set_index(replace_with_warning(hfa_newest["Country name"],any_to_wb))
# READ THE PREVIOUS HFA DATA
hfa_previous=pd.read_csv(inputs+"HFA_all_2009_2011.csv")
hfa_previous=hfa_previous.set_index(replace_with_warning(hfa_previous["Country name"],any_to_wb))
#most recent values... if no 2011-2013 reporting, we use 2009-2011
hfa_oldnew=pd.concat([hfa_newest, hfa_previous, hfa15], axis=1,keys=['new', 'old', "15"]) #this is important to join the list of all countries
hfa = hfa_oldnew["15"].fillna(hfa_oldnew["new"].fillna(hfa_oldnew["old"]))
hfa["shew"]=1/5*hfa["P2-C3"] #access to early warning normalized between zero and 1.
hfa["prepare_scaleup"]=(hfa["P4-C2"]+hfa["P5-C2"]+hfa["P4-C5"])/3/5 # q_s in the report, ability to scale up support to to affected population after the disaster, normalized between zero and 1
hfa["finance_pre"]=(1+hfa["P5-C3"])/6 #betwenn 0 and 1	!!!!!!!!!!!!!!!!!!!REMARK: INCONSISTENT WITH THE TECHNICAL PAPER. Q_f=1/2(ratings+P5C3/5)
df[["shew","prepare_scaleup","finance_pre"]]=hfa[["shew","prepare_scaleup","finance_pre"]]
df[["shew","prepare_scaleup","finance_pre"]]=df[["shew","prepare_scaleup","finance_pre"]].fillna(0)	#assumes no reporting is bad situation (caution! do the fillna after inputing to df to get the largest set of index)

###Income group
#df["income_group"]=pd.read_csv(inputs+"income_groups.csv",header=4,index_col=2)["Income group"].dropna()

###Country Ratings
the_credit_rating_file=inputs+"credit_ratings_scrapy.csv"
if use_2016_inputs or use_2016_ratings: the_credit_rating_file=inputs+"cred_rat.csv"

nb_weeks=(time.time()-os.stat(the_credit_rating_file).st_mtime )/(3600*24*7)
if nb_weeks>3:
    warnings.warn("Credit ratings are "+str(int(nb_weeks))+" weeks old. Get new ones at http://www.tradingeconomics.com/country-list/rating")
    #assert(False)

ratings_raw=pd.read_csv(the_credit_rating_file,dtype="str", encoding="utf8").dropna(how="all") #drop rows where only all columns are NaN.
ratings_raw=ratings_raw.rename(columns={"Unnamed: 0": "country_in_ratings"})[["country_in_ratings","S&P","Moody's","Fitch"]]	#Rename "Unnamed: 0" to "country_in_ratings" and pick only columns with country_in_ratings, S&P, Moody's and Fitch.
ratings_raw.country_in_ratings= ratings_raw.country_in_ratings.str.strip().replace(["Congo"],["Congo, Dem. Rep."])	#The creidt rating sources calls DR Congo just Congo. Here str.strip() is needed to remove any space in the raw data. In the raw data, Congo has some spaces after "o". If not used str.strip(), nothing is replaced.
ratings_raw["country"]= replace_with_warning(ratings_raw.country_in_ratings.apply(str.strip),any_to_wb)	#change country name to wb's name

ratings_raw=ratings_raw.set_index("country")
ratings_raw=ratings_raw.applymap(mystriper)	#mystriper is a function in lib_gather_data. To lower case and strips blanks.

#Transforms ratings letters into 1-100 numbers
rat_disc = pd.read_csv(inputs+"cred_rat_dict.csv")
ratings=ratings_raw
ratings["S&P"].replace(rat_disc["s&p"].values,rat_disc["s&p_score"].values,inplace=True)
ratings["Moody's"].replace(rat_disc["moodys"].values,rat_disc["moodys_score"].values,inplace=True)
ratings["Fitch"].replace(rat_disc["fitch"].values,rat_disc["fitch_score"].values,inplace=True)
ratings["rating"]=ratings.mean(axis=1)/100 #axis=1 is the average across columns, axis=0 is to average across rows. .mean ignores NaN
df["rating"] = ratings["rating"]
if True:
    print("some bad rating occurs for" + "; ".join(df.loc[isnull(df.rating)].index))
df["rating"].fillna(0,inplace=True)  #assumes no rating is bad rating

###Ratings + HFA
df["borrow_abi"]=(df["rating"]+df["finance_pre"])/2 # Ability and willingness to improve transfers after the disaster

###If contingent finance instrument then borrow_abo = 1
contingent_file=inputs+"contingent_finance_countries.csv"
if use_2016_inputs: contingent_file=inputs+"Contingent_finance_countries_orig.csv"
which_countries=pd.read_csv(contingent_file,dtype="str", encoding="utf8").set_index("country")
which_countries["catDDO"]=1
df = pd.merge ( df.reset_index() , which_countries.reset_index() , on = "country" , how="outer").set_index("country")
df.loc[df.catDDO==1,"borrow_abi"]=1
#df.loc[df.catDDO==1,"prepare_scaleup"]=1

#if True: df['borrow_abi'] = 2
df = df.drop(["catDDO"],axis =1 )

print(df)


##Capital data
k_data=pd.read_csv(inputs+"capital_data.csv", usecols=["code","cgdpo","ck"]).replace({"ROM":"ROU","ZAR":"COD"}).rename(columns={"cgdpo":"prod_from_k","ck":"k"})#Zair is congo
iso_country = pd.read_csv(inputs+"iso3_to_wb_name.csv", index_col="iso3")	#matches names in the dataset with world bank country names
k_data.set_index("code",inplace=True)
k_data["country"]=iso_country["country"]
cond = k_data["country"].isnull()
if cond.sum()>0:
     warnings.warn("this countries appear to be missing from iso3_to_wb_name.csv: "+" , ".join(k_data.index[cond].values))
k_data=k_data.reset_index().set_index("country")
df["avg_prod_k"]=k_data["prod_from_k"]/k_data["k"]	#\mu in the technical paper -- average productivity of capital

#####
#for SIDS, adding capital data from GAR
sids_k = pd.read_csv("intermediate/avg_prod_k_with_gar_for_sids.csv").rename(columns={"Unnamed: 0":"country"}).set_index("country")
df = df.fillna(sids_k)
df.dropna().shape
#####

##Hazards data
###Vulnerability from Pager data
pager_description_to_aggregate_category = pd.read_csv(inputs+"pager_description_to_aggregate_category.csv", index_col="pager_description", squeeze=True)
PAGER_XL = pd.ExcelFile(inputs+"PAGER_Inventory_database_v2.0.xlsx")
pager_desc_to_code = pd.read_excel(PAGER_XL,sheet_name="Release_Notes", usecols="B:C", skiprows=56).dropna().squeeze()
pager_desc_to_code.Description = pager_desc_to_code.Description.str.strip(". ")	#removes spaces and dots from PAGER description
pager_desc_to_code.Description = pager_desc_to_code.Description.str.replace("  "," ")	#replace double spaces with single spaces
pager_desc_to_code = pager_desc_to_code.set_index("PAGER-STR")
pager_code_to_aggcat = replace_with_warning( pager_desc_to_code.Description, pager_description_to_aggregate_category, joiner="\n") #results in a table with PAGER-STR index and associated category (fragile, median etc.)

###total share of each category of building per country
rural_share= .5*get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheet_name='Rural_Non_Res')+.5*get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheet_name='Rural_Res')
urban_sare = .5*get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheet_name='Urban_Non_Res')+.5*get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheet_name='Urban_Res')
share = (rural_share.stack()*(1-df.urbanization_rate) + urban_sare.stack()*df.urbanization_rate).unstack().dropna()	#the sum(axis=1) of rural_share is equal to 1, so rural_share needs to be weighted by the 1-urbanization_rate, same for urban_share
share=  share[share.index.isin(iso3_to_wb)] #the share of building inventory for fragile, median and robust

###matching vulnerability of buildings and people's income and calculate poor's, rich's and country's vulnerability
agg_cat_to_v = pd.read_csv(inputs+"aggregate_category_to_vulnerability.csv", sep=";", index_col="aggregate_category", squeeze=True)
##REMARK: NEED TO BE CHANGED....Stephane I've talked to @adrien_vogt_schilb and don't want you to go over our whole conversation. Here is the thing: in your model, you assume that the bottom 20% of population gets the 20% of buildings of less quality. I don't think it's a fair jusfitication, because normally poor people live in buildings of less quality but in a more crowded way,i.e., it could be the bottom 40% of population get the 10% of buildings of less quality. I think we need to correct this matter. @adrien_vogt_schilb also agreed on that, if he didn't change his opinion. How to do that? I think once we incorporate household data, we can allocate buildings on the decile of households, rather than population. I think it's a more realistic assumption.
p=(share.cumsum(axis=1).add(-df["pov_head"],axis=0)).clip(lower=0)
poor=(share-p).clip(lower=0)
rich=share-poor
vp_unshaved=((poor*agg_cat_to_v).sum(axis=1, skipna=False)/df["pov_head"] )
vr_unshaved=(rich*agg_cat_to_v).sum(axis=1, skipna=False)/(1-df["pov_head"])
v_unshaved =  vp_unshaved*df.share1 + vr_unshaved*(1-df.share1)
v_unshaved.name="v"
v_unshaved.index.name = "country"

vp = vp_unshaved.copy()
vr = vr_unshaved.copy()
v = v_unshaved.copy()

###apply \delta_K = f_a * V, and use destroyed capital from GAR data, and fa_threshold to recalculate vulnerability
frac_value_destroyed_gar = pd.read_csv(intermediate+"frac_value_destroyed_gar_completed.csv", index_col=["country", "hazard", "rp"], squeeze=True);#\delta_K, Generated by pre_process\ GAR.ipynb

fa_guessed_gar = ((frac_value_destroyed_gar/broadcast_simple(v_unshaved,frac_value_destroyed_gar.index)).dropna()).to_frame()
#fa is the fraction of asset affected. broadcast_simple, substitute the value in frac_value_destroyed_gar by values in v_unshaved.
# Previously, it assumed that vulnerability for all types of disasters are the same. fa_guessed_gar = exposure/vulnerability
# Now we will change this to event-specific vulnerabilities...
fa_guessed_gar.columns = ['fa']


# merge v with hazard_ratios
fa_guessed_gar = pd.merge(fa_guessed_gar.reset_index(),vr.reset_index(),on=economy)
fa_guessed_gar = pd.merge(fa_guessed_gar.reset_index(),vp.reset_index(),on=economy).drop('index',axis=1)
fa_guessed_gar.columns = ['country','hazard','rp','fa','nonpoor','poor']

# stack and get columns right
try: fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level+['fa']).drop(['index'],axis=1).stack()
except: fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level+['fa']).stack()
fa_guessed_gar.index.names = event_level+['fa','income_cat']
fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level+['income_cat'])
fa_guessed_gar.columns = ['fa','v']

excess=fa_guessed_gar.loc[fa_guessed_gar.fa>fa_threshold,'fa']
for c in excess.index:
    r = (excess/fa_threshold)[c]
    print('\nExcess case:\n',c,'\n',r,'\n',fa_guessed_gar.ix[c],'\n\n')
    fa_guessed_gar.loc[c,'fa'] = fa_guessed_gar.loc[c,'fa']/r  # i don't care.
    fa_guessed_gar.loc[c,'v'] = fa_guessed_gar.loc[c,'v']*r
    print('r=',r)
    print('changed to:',fa_guessed_gar.loc[c],'\n\n')

fa_guessed_gar['v'] = fa_guessed_gar['v'].clip(upper=0.99)
fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level)

###Exposure bias from PEB
data = pd.read_excel(inputs+"PEB_flood_povmaps.xlsx")[["iso","peb"]].dropna()	#Exposure bias from WB povmaps study
df["pe"] = data.set_index(data.iso.replace(iso3_to_wb)).peb-1
PEB_wb_deltares_older = pd.read_csv(inputs+"PEB_wb_deltares.csv",skiprows=[0,1,2],usecols=["Country","Nation-wide"])	#Exposure bias from older WB DELTARES study
PEB_wb_deltares_older["country"] = replace_with_warning(PEB_wb_deltares_older["Country"],any_to_wb) #Replace with warning is used for columns, for index set_index is needed.
df["pe"]=df["pe"].fillna(PEB_wb_deltares_older.set_index("country").drop(["Country"],axis=1).squeeze()) #Completes with bias from previous study when pov maps not available. squeeze is needed or else it's impossible to fillna with a dataframe
if use_avg_pe:
    df["pe"]=df["pe"].fillna(wavg(df["pe"],df["pop"])) #use averaged pe from global data for countries that don't have PE.
else:
    df["pe"].fillna(0)
pe = df.pop("pe")

###incorporates exposure bias, but only for (riverine) flood and surge, and gets an updated fa for income_cats
#fa_hazard_cat = broadcast_simple(fa_guessed_gar,index=income_cats) #fraction of assets affected per hazard and income categories
#fa_with_pe = concat_categories(fa_guessed_gar*(1+pe),fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head), index=income_cats)
## NB: fa_guessed_gar*(1+pe) gives f_p^a and fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head) gives f_r^a. TESTED
#fa_guessed_gar.update(fa_with_pe) #updates fa_guessed_gar where necessary

fa_with_pe = fa_guessed_gar.query("hazard in ['flood','surge']")[['income_cat','fa']].copy()#selects just flood and surge
fa_with_pe.loc[fa_with_pe['income_cat']=='poor','fa'] = fa_with_pe.loc[fa_with_pe['income_cat']=='poor','fa']*(1+pe)
fa_with_pe.loc[fa_with_pe['income_cat']=='nonpoor','fa'] = fa_with_pe.loc[fa_with_pe['income_cat']=='nonpoor','fa']*(1-df.pov_head*(1+pe))/(1-df.pov_head)

fa_with_pe = fa_with_pe.reset_index().set_index(event_level+['income_cat'])
fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level+['income_cat'])

fa_guessed_gar['fa'].update(fa_with_pe['fa'])
if constant_fa:
    if use_2016_inputs: fa_guessed_gar.to_csv(inputs+'constant_fa.csv',header=True)
    else:
        fa_guessed_gar['fa'].update(pd.read_csv('orig_inputs/constant_fa.csv',index_col=['country','hazard','rp','income_cat'])['fa'])

###gathers hazard ratios
hazard_ratios = pd.DataFrame(fa_guessed_gar)
hazard_ratios = pd.merge(hazard_ratios.reset_index(),df['shew'].reset_index(),on=['country']).set_index(event_level+['income_cat'])
#hazard_ratios["shew"]=broadcast_simple(df.shew, index=hazard_ratios.index)
hazard_ratios["shew"]=hazard_ratios.shew.unstack("hazard").assign(earthquake=0).stack("hazard").reset_index().set_index(event_level+[ "income_cat"]) #shew at 0 for earthquake
if not no_protection:
    #protection at 0 for earthquake and wind
    hazard_ratios["protection"]=1
    hazard_ratios["protection"]=hazard_ratios.protection.unstack("hazard").assign(earthquake=1, wind=1).stack("hazard").reset_index().set_index(event_level)
hazard_ratios= hazard_ratios.drop("Finland") #because Finland has fa=0 everywhere.

##Protection
if protection_from_flopros: #in this code, this protection is overwritten by no_protection
    minrp = 1/2 #assumes nobody is flooded more than twice a year
    df["protection"]= pd.read_csv(inputs+"protection_national_from_flopros.csv", index_col="country", squeeze=True).clip(lower=minrp)
else: #assumed a function of the income group
    protection_assumptions = pd.read_csv(inputs+"protection_level_assumptions.csv", index_col="Income group", squeeze=True)
    df["protection"]=pd.read_csv(inputs+"income_groups.csv",header =4,index_col=2)["Income group"].dropna().replace(protection_assumptions)
if no_protection:
    p=hazard_ratios.reset_index("rp").rp.min()
    df.protection=p
    print("PROTECTION IS ",p)

##Data by income categories
cat_info =pd.DataFrame()
cat_info["n"]  = concat_categories(ph,(1-ph),index= income_cats)	#number
cp=   df["share1"] /ph    *df["gdp_pc_pp"]	#consumption levels, by definition.
cr=(1-df["share1"])/(1-ph)*df["gdp_pc_pp"]
cat_info["c"]       = concat_categories(cp,cr,index= income_cats)
cat_info["social"]  = concat_categories(df.social_p,df.social_r,index= income_cats)	#diversification
cat_info["axfin"] = concat_categories(df.axfin_p,df.axfin_r,index= income_cats)	#access to finance
cat_info = cat_info.dropna()

##Taxes, redistribution, capital
df["tau_tax"],cat_info["gamma_SP"] = social_to_tx_and_gsp(economy,cat_info)	#computes tau tax and gamma_sp from socail_poor and social_nonpoor. CHECKED!
cat_info["k"] = (1-cat_info["social"])*cat_info["c"]/((1-df["tau_tax"])*df["avg_prod_k"]) #here k in cat_info has poor and non poor, while that from capital_data.csv has only k, regardless of poor or nonpoor

#Exposure
cat_info["fa"] =hazard_ratios.fa.mean(level=["country","income_cat"])

#Vulnerability
#cat_info["v"] = concat_categories(vp,vr, index=income_cats)
# NB: vulnerability has been moved to hazard_ratios, and i'll leave it out here to avoid bugs later.

#access to early warnings
cat_info["shew"] = hazard_ratios.shew.drop("earthquake", level="hazard").mean(level=["country","income_cat"])

_df            = df.copy('deep')
_cat_info      = cat_info.copy('deep')
_hazard_ratios = hazard_ratios.copy('deep')

#######

#print "OK up to here"

# Create loop over policies
#for apol in [None]:
for apol in [None, ['bbb_complete',1],['borrow_abi',2], 'unif_poor', ['bbb_incl',1], ['bbb_fast',1], ['bbb_fast',2], ['bbb_fast',4], ['bbb_fast',5], ['bbb_50yrstand',1]]:

    pol_opt = None
    try:
        pol_opt = apol[1]
        pol_str = apol[0]
    except:
        pol_str = apol

    # apply policy apol
    df,cat_info,hazard_ratios,a,desc=apply_policy(_df,_cat_info,_hazard_ratios,pol_str,pol_opt)

    # clean up and save out
    if drop_unused_data:
        cat_info= cat_info.drop([i for i in ["social"] if i in cat_info.columns],axis=1, errors="ignore").dropna()
        df_in = df.drop([i for i in ["social_p", "social_r","pov_head", "pe","vp","vr", "axfin_p",  "axfin_r","rating","finance_pre"] if i in df.columns],axis=1, errors="ignore").dropna()
    else :
        df_in = df.dropna()
    df_in = df_in.drop([ "shew"],axis=1, errors="ignore").dropna()

    #Save all data
    print(df_in.shape[0],'countries in analysis')

    try:
        pol_str = '_'+pol_str+str(pol_opt)
    except:
        pol_str = ''

    fa_guessed_gar.to_csv(intermediate+"/fa_guessed_from_GAR_and_PAGER_shaved"+pol_str+".csv",encoding="utf-8", header=True)
    pd.DataFrame([vp,vr,v], index=["vp","vr","v"]).T.to_csv(intermediate+"/v_pr_fromPAGER_shaved_GAR"+pol_str+".csv",encoding="utf-8", header=True)
    df_in.to_csv(intermediate+"/macro"+pol_str+".csv",encoding="utf-8", header=True)
    cat_info.to_csv(intermediate+"/cat_info"+pol_str+".csv",encoding="utf-8", header=True)
    hazard_ratios.to_csv(intermediate+"/hazard_ratios"+pol_str+".csv",encoding="utf-8", header=True)
