# Europe_flood_road_disruption
Percolation analysis for the European road network with flood hazard data. This code is used to create the results of the paper <i>Will river floods tip European road networks?</i>, by Kees van Ginkel, Elco Koks, Frederique de Groen, Viet Dung Nguyen and Lorenzo Alfieri (2021, under review).

# Code overview
Note that the paths are set in the config.json file, which is loaded in utils.py.<br />
Besides these generic utils, there are some specific functions for handling data on the EU (NUTS) classification, in Europe_utils.py

### Step 0: Preprocessing with TRAILS (outside this Github)
This step is done outside this Github project. First, the python package TRAILS is used to prepare clean graphs (iGraph objects) from an OpenStreetMap planet dump. This is done for individual countries. The network of each country is saved as an edges.feather (vertices) and nodes.feather file. This file format can be easily read with (Geo)Pandas. <br />
Second, the EFAS/LISFLOOD flood hazard data (100 m resolution) is intersected with the network edges. This adds the flood depth (m) per return period (e.g. 1:100 year event) to the network edges; i.e. the road segments in the network.

### Step 1: Preprocessing 
This step first calculates the centroids of all NUTS-2 and NUTS-3 regions. It then calculates for each country, the optimal routes between all NUTS-2 or NUTS-3 regions. The results of this step are saved in the preproc_output folder.<br />

To complete the preprocessing step, run the following scripts:<br />
 - <i>preprocessing.py / create_centroids_csv():</i> this finds the centroids for all NUTS2 and NUTS3 regions.
 - <i>preprocessing.py / optimal_routes()</i> this creates the optimal routes between all NUTS2 and NUTS3 region.

Two import side-products of this step are: 
 - the scheduled nuts2/3_combinations.csv files, which determine the different number of AoIs that will be sampled in the next step; settings in this file may be overruled when the preprocessing is finished
 - a file showing the optimal (preferred) routes between the NUTS2/3-regions; these are a good quality check of the graphs and routing functionality
Inspect these files before going to the next step.<br /><br />

### Step 2: Percolation analysis
In the second step, the actual percolation analysis is done. In short, the sampling procedure is read from the file <i> data/nuts3_combinations </i>, and the result are unaggregated csv files with the results of the percolation analysis, written to the <i>main_output</i> folder. <br/>

In more detail, the percolation analysis is coordinated from the script <i>run_percolation_parallel.py</i>, which calls functionality from <i>percolation_optimized_parallel.py</i>. The percolation analysis is run in two phases to enable parallel processing.<br/><br/>
In phase 1 (<i>called with prep_par() in run_percolation_parallel.py</i>) all experiments that are to be done are scheduled. Scheduled experiments are written to <i>main_output / {country} / scheduled</i> in pickle <i>.pkl</i> format. 
In phase 2 (<i>called with run_par() in run_percolation_parallel.py</i>) the actual experiments are carried out, and the results are wriiten to  <i>main_output / {country} / finished</i>. The main benefit of running phase 2 seperate from phase 1 is that this enables running phase2 parallel on multiple machines at the same time, provided they have access and read/write from the same <i>main output</i> folder. 

### Step 3: Postprocessing and visualisation
When the experiments are finished, the results need to be merged into a single 'all_combinations.csv' file in the folder <i>main_output</i>, using the script <i>run_aggregation.py</i>. <br />
Having aggregated the results, there are countless ways to visualise the results, we name a few of the most common and their required data manipulations. The core code for visualisations can be found in <i>visualisations.py</i>. Around this core code, we provide many Jupyter notebooks which deliver annoted and styled plots, including those that can be found in paper (Van Ginkel et al, 2021, under review).<br /><br />

The function <i>visualisations.py -> main()</i> Reads the (merged) outputs of the percolation analysis from the csv file in the folder <i>main_output / 'all_combinations.csv'</i>, and does some basic assembly work which is useful for many visualisations. It returns a Pandas dataframe containing the raw results, a datafrmame with results grouped by  absolute #AoIs AND country, and a dataframe with results grouped by relative #AoIs AND country. The aggregated dataframes only cover the metric <b>preferred routes disrupted</b>, in short 'disrupted'.<br >
To obtain relative and absolute aggregated results for the other metrics, run <i>process_no_detour()</i> for <b>isolated trips</b> and <i>process_extra_time()</i> for <i>travel time increase.

