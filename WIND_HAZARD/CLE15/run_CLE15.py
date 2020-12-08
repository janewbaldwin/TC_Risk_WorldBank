# 1) run CLE15.py

# 2) run the following
### constant from Dan Chavas ###
fcor = 5.e-5;                    #[s-1] {5e-5}; Coriolis parameter at storm center
# Environmental parameters
##Outer region
Cdvary = 1;                     #[-] {1}; 0 : Outer region Cd = constant (defined on next line); 1 : Outer region Cd = f(V) (empirical Donelan et al. 2004)
Cd = 1.5e-3;                    #[-] {1.5e-3}; ignored if Cdvary = 1; surface momentum exchange (i.e. drag) coefficient
w_cool = 2./1000;                #[ms-1] {2/1000; Chavas et al 2015}; radiative-subsidence rate in the rain-free tropics above the boundary layer top

##Inner region
CkCdvary = 1;                   #[-] {1}; 0 : Inner region Ck/Cd = constant (defined on next line); 1 : Inner region Ck/Cd = f(Vmax) (empirical Chavas et al. 2015)
CkCd = 1.;                       #[-] {1}; ignored if CkCdvary = 1; ratio of surface exchange coefficients of enthalpy and momentum; capped at 1.9 (things get weird >=2)

## Eye adjustment
eye_adj = 0;                    #[-] {1}; 0 = use ER11 profile in eye; 1 = empirical adjustment
alpha_eye = .15;                #[-] {.15; empirical Chavas et al 2015}; V/Vm in eye is reduced by factor (r/rm)^alpha_eye; ignored if eye_adj=0
###

rr,VV,r0,rmerge,Vmerge = CLE15.ER11E04_nondim_rmaxinput(required input)
