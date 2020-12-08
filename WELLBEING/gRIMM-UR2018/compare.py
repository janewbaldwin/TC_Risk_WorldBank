from res_ind_lib import *
from lib_compute_resilience_and_risk import *

from replace_with_warning import *
import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_old = pd.read_csv(os.getcwd()+'/output/results_tax_unif_poor_ copy.csv')
df_new = pd.read_csv(os.getcwd()+'/output/results_tax_unif_poor_.csv')

df_sth = pd.read_excel(os.getcwd()+'/../stephane_results_tax_unif_poor_.xlsx')
_df = pd.merge(df_sth.reset_index(),df_new.reset_index(),on='country')

#df_book = pd.read_excel(os.getcwd()+'/results/results_comparison.xlsx',skiprows=1)
#_df = pd.merge(df_book[['country','risk','resilience','risk_to_assets']].reset_index(),df_new.reset_index(),on='country')


print(df_old.shape[0],'countries in df_old. dKtot =',df_old.dKtot.sum())
print(df_new.shape[0],'countries in df_new: dKtot =',df_new.dKtot.sum())
print(round(100.*(df_new.dKtot.sum()-df_old.dKtot.sum())/df_old.dKtot.sum(),1),'% increase\n')

print('\nComparing results for',_df.shape[0],'countries:\n')

_df['RA_change'] = (_df['risk_to_assets_y']-_df['risk_to_assets_x'])/_df['risk_to_assets_x']
_df.plot.scatter('risk_to_assets_x','RA_change')
plt.show()

_df.loc[_df.country=='Philippines'].squeeze().to_csv('~/Desktop/_df.csv')
print(_df[['country','RA_change','risk_to_assets_y','risk_y','resilience_x','resilience_y']].sort_values('RA_change',ascending=True))

_df.plot.scatter('risk_to_assets_x','risk_to_assets_y')
plt.show()

print(_df.columns)

_df['RES_change'] = (_df['resilience_y']-_df['resilience_x'])/_df['resilience_x']

plt.cla()
_df.plot.scatter('resilience_x','RES_change')
plt.show()

plt.close('all')
