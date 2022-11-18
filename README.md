## Replication code for "Deep Learning to Map Concentrated Animal Feeding Operations"
The code and data in this repository will replicate the tables and figures in
the main body of "Deep Learning to Map Concentrated Animal Feeding Operations."
The replication code is designed to work on macOS.

### Attribution
When using these models and/or the underlying data, please cite the paper above
with mention that the location data used to train the models was developed by
the [Environmental Working Group](https://www.ewg.org/) and
[Waterkeeper Alliance](https://waterkeeper.org/), with validation by the
research team at Stanford.

### Requirements
1. The python code requires the dependencies in the cafo_environment.yml file.
To replicate this environment, install the
[open-source anaconda distribution](https://www.anaconda.com/distribution/),
navigate to the top level of the replication directory, and run
```
conda env create --name cafo -f=environments/cafo_environment.yml
```
2. The R code was developed using the session described in
environments/r_session_info. Upon execution, the R scripts will
automatically to try to install the required packages and their dependencies.
3. Figure 5 requires a Google Maps Static API key, which the script will ask for.
Instructions on how to obtain one are provided in the
[Google Maps Platform documentation](https://developers.google.com/maps/documentation/maps-static/get-api-key).

### Naming conventions
Each script that creates a table or a figure is named for the figure or table
it creates in the manuscript as it appears in the raw tex file. The naming
convention is `[Item number]_[item_type]_[item name]`.
The script `create_cafo_facilities.py` replicates the facility consolidation
algorithm that was run after the class activation mapping and rescoring process
demonstrated in `3_Figure_fgcam_CAM_Algorithm_Illustration.py.`

### Instructions
To replicate all figures and tables, open a bash shell and activate the
python virtual environment created from cafo_environment.yml. Navigate to the
top level of the replication directory and run:
```
chmod u+x code/replicate_all.sh
./replicate_all.sh
```
The variable names and descriptions of all variables in the csv files are
included in data/csv/codebook.xlsx.
