# TC_Risk_WorldBank
SUMMARY:
This is code to support the paper "Vulnerability in a Tropical Cyclone Risk Model: Philippines Case Study" to be published in the journal Weather, Climate, and Society. Data to support this project can be found on the DesignSafe platform doi:XXXXXXXXXXX. This project produced a tropical cyclone risk (e.g. asset loss) model for the Philippines based on open-source data, geared at supporting nonprofit applications. The files and code included in this data repository allow for reproduction of the different model components (e.g. the hazard, vulnerability, and exposure layers) and the model validation. The data produced by this model and used in the publication are also made available here-- namely the tropical cyclone wind fields for observed and synthetic storms over the Philippines, the vulnerability and exposure layers, and the resultant asset losses. 


INSTRUCTIONS FOR USAGE:

The code is written in Python in the form of Jupyter Notebooks or scripts. 

To run the code first install the relevant Python environments included in the worldbank.yml and geopandas.yml files. Using Anaconda this can be done as follows:
conda env create -f worldbank.yml
conda env create -f geopandas.yml
The worldbank environment is used for almost all of this project. The only exception is the notebooks in /REGION_MASKS which should be run in the geopandas environment.





