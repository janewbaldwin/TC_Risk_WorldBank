# Scripts
* `phi_1vcurve_sensitivity.py`: Calculates asset losses for historical Philippines landfalling swaths based on IBTrACS, assuming the same vulnerability curve for everywhere in the Philippines. Results are calculated for a range of different vulnerability parameter values (Vthresh = 15-35 m/s by 5, and Vhalf = 50-200 m/s by 10). Output from this script is large (each file is ~6GB totaling 100s of GB).
* `region_vulnerability_fit.py`:  Estimates metrics of asset loss simulation accuracy (TDR and RMSF) by comparing asset losses from `phi_1vcurve_sensitivity.py` to EM-DAT observed losses. Comparison is conducted for the range of individual vulnerability curves tested, for the entire Philippines using all Philippines-landfalling storms, and for subsets of storms making landfall in each region. Output from this is used as input to `/VULNERABILITY/Vulnerability_Curves_Params_from_Household_Survey_Data.ipynb` to produce regional vulnerability maps.
* `assetlosses_ibtracs_vulnerabilitybyregion.ipynb`: Produces asset losses for historical Philippines landfalling swaths based on IBTrACS, using the regional vulnerability map from `/VULNERABILITY/Vulnerability_Curves_Params_from_Household_Survey_Data.ipynb`. Creates right panel of Figure 4.
* `assetlosses_chaz_vulnerabilitybyregion.py`: Produces asset losses for historical Philippines landfalling swaths based on CHAZ tracks, using the regional vulnerability map from `/VULNERABILITY/Vulnerability_Curves_Params_from_Household_Survey_Data.ipynb`. Since this is a large amount of data, only regional quantities are saved out-- regional maximum wind speed and regionally summed asset losses.
* `emdat_comparison_phi.ipynb`: Compares asset losses for historical Philippines landfalling swaths based on IBTrACS to EM-DAT observed losses, using national vs regional-fit vulnerability. Produces Figures 7, 8, 11, 12, and S4.
* `exceedance_curves.ipynb`: Creates Figures 14 and 15.
* `figures_for_presentations.ipynb`: Creates Figures 1 left panel, 5, and 13.

# Referenced data
Note: all data used by these scripts is in the DesignSafe repository, and the below links for provided just for reference.
* EM-DAT: The International Disaster Database | https://www.emdat.be/database
* LitPop: Global Exposure Data for Disaster Risk Assessment | https://doi.org/10.3929/ethz-b-000331316
* Penn World Table version 10.01 | https://doi.org/10.34894/QT5BCC
