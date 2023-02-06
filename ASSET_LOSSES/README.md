# Scripts
* `ibtracs_windfields_phi.py`: Creates wind swaths from the historical IBTrACS tracks for Philippines-landfalling storms.
* `synthetic_data_windfields_phi.py`:  Creates wind swaths from the syntehtic CHAZ tracks for Philippines-landfalling storms. CHAZ tracks are downscaled from ERA-Interim historical reanalysis data.
* `run_synthetic_phi.py`: Runs `synthetic_data_windfields_phi.py` over the 7 sets of CHAZ tracks (numbers 000-007).
* `windfield_validation.ipynb`: Compares simulated Philippines windfields to those from observational SAROPS windfield product. Creates Figures S2 and S3.
* `windswath_validation.ipynb`: Compares simulated US swaths to those from observational HWIND wind swath product. Creates Figure S1.

# Referenced data
Note: all data used by these scripts is in the DesignSafe repository, except for the HWIND data which must be downloaded from the below link.
* International Best Track Archive for Climate Stewardship (IBTrACS) | doi:10.25921/82ty-9e16
* SAROPS Tropical Cyclone Winds | https://www.star.nesdis.noaa.gov/socd/mecb/sar/AKDEMO_products/APL_winds/tropical/index.html
* HWIND Legacy Archive | https://www.rms.com/event-response/hwind/legacy-archive

# Related project
Columbia Tropical Cyclone Hazard model (CHAZ) | https://github.com/cl3225/CHAZ
