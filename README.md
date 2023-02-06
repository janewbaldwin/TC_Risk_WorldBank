# TC_Risk_WorldBank
## SUMMARY:
This is code to support the paper Baldwin et al 2023 ["Vulnerability in a Tropical Cyclone Risk Model: Philippines Case Study"](https://essopenarchive.org/doi/full/10.1002/essoar.10511053.1) to be published in the journal *Weather, Climate, and Society*. Data to support this project can be downloaded from the DesignSafe platform doi:XXXXXXXXXXX. 

This project produced a tropical cyclone risk (e.g. asset loss) model for the Philippines based on open-source data, geared at supporting nonprofit applications. The files and code included in this data repository allow for reproduction of the different model components (e.g. the hazard, vulnerability, and exposure layers) and the model validation. The data produced by this model and used in the publication are also made available here-- namely the tropical cyclone wind fields for observed and synthetic storms over the Philippines, the vulnerability and exposure layers, and the resultant asset losses. 


## INSTRUCTIONS FOR USE:

The code is written in Python in the form of Jupyter Notebooks and scripts. 

To reproduce the paper results:

1) Download/clone the data and code for this project onto your personal system.

2) Install the relevant Python environments included in the worldbank.yml and geopandas.yml files. Using Anaconda this can be done as follows:
conda env create -f worldbank.yml
conda env create -f geopandas.yml
The worldbank environment is used for almost all of this project. The only exception is the notebooks in /REGION_MASKS which should be run in the geopandas environment.

3) Change root_dir at the beginning of each script to the path to where you downloaded the project data. Initially this is set as: root_dir = '/data2/jbaldwin/WCAS2023'.

4) Add the following line to your ~/.bashrc file, replacing /home/jbaldwin/WCAS2023 with the location of this code on your system:
export PYTHONPATH=/home/jbaldwin/WCAS2023/FUNCTIONS
This allows you to import the custom library of functions that support this work.


## CODE/DATA ORGANIZATION

The code and data is organized into directories based on its purpose. Brief descriptions of the directory contents are below. 
Please refer to README files within each directory for details on what each script and function does, and the relevant paper figures it produces.

* /FUNCTIONS: custom functions that support this work.
* /REGION_MASKS: creates Philippines region masks to apply in later analysis of wind swaths and asset losses.
* /EXPOSED_VALUE: converts the exposed value LitPop dataset into a netcdf form.
* /HAZARD: produces wind swaths from tropical cyclone tracks (IBTrACS and CHAZ), and validates the swaths.
* /ASSET_LOSSES: estimates asset losses from wind swaths for various vulnerability levels, and compares to EM-DAT observed losses.
* /VULNERABILITY: calibrates vulnerability for regions by comparing to Philippines survey data.

These directories are listed in general order of how they should be run to reproduce all the work in this paper-- the one exception being that some of the code in /ASSET_LOSSES depends on results from /VULNERABILITY. However, data is provided on DesignSafe to support intermediate stages of analysis, so individual scripts can be also be run separately.


## CITATION INSTRUCTIONS:

Please cite this project as:

Jane W. Baldwin, Chia-Ying Lee, Brian Walsh, Suzana Camargo, and Adam Sobel. Vulnerability in a Tropical Cyclone Risk Model: Philippines Case Study. *ESS Open Archive*. April 08, 2022.
[DOI: 10.1002/essoar.10511053.1](https://essopenarchive.org/doi/full/10.1002/essoar.10511053.1)**

**this will soon be replaced by the citation of the final publication in *Weather, Climate, and Society*.

For details on the Columbia Tropical Cyclone Hazard model (CHAZ) and proper attribution of that product see: https://github.com/cl3225/CHAZ
Synthetic storms from this model are analyzed in this paper.






