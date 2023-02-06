# TC_Risk_WorldBank
## SUMMARY:
This is code to support the paper "Vulnerability in a Tropical Cyclone Risk Model: Philippines Case Study" to be published in the journal Weather, Climate, and Society. Data to support this project can be found on the DesignSafe platform doi:XXXXXXXXXXX. This project produced a tropical cyclone risk (e.g. asset loss) model for the Philippines based on open-source data, geared at supporting nonprofit applications. The files and code included in this data repository allow for reproduction of the different model components (e.g. the hazard, vulnerability, and exposure layers) and the model validation. The data produced by this model and used in the publication are also made available here-- namely the tropical cyclone wind fields for observed and synthetic storms over the Philippines, the vulnerability and exposure layers, and the resultant asset losses. 


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

The code and data is organized into directories based on its purpose. Brief descriptions of the directory contents are below. Please refer to README files within each directory for details on what each script and function does, and the relevant paper figures it produces.

* /FUNCTIONS: custom functions that support this work.
* 







