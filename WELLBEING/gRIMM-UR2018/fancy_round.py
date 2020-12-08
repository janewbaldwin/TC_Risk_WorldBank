import numpy as np

def fancy_round(x,n=2):
    """displays numbers with a given number of total digits

    fancy_round(12679.30450,2)
    13000
    fancy_round(12.30450,2)
    12
    fancy_round(0.000545,2)
    0.0
    """
    
    #special cases
    if np.isnan(x):
        return x
    
    if np.isinf(x):
        return (x)
    
    #number of digits before the dot in the original number
    if abs(x)<1:
        s=0
    else:
        s=int(np.log10(abs(x)))+1
    
    #output
    if s<n:
        return round(x,n-s)
    else:
        return int(round(x,n-s))
        
        
from math import log10, floor
def round_sig(x, sig=2):
    """keeps only 2 significant digits in x
    round_sig(12679.30450,2)
    13000.0
    round_sig(12.30450,2)
    12.0
    round_sig(0.000545,2)
    0.00055
    """     
        
    if np.isnan(x) or np.isinf(x) or (x==0):
        return x
    return round(x, sig-int(floor(log10(abs(x)))-1)-2)        