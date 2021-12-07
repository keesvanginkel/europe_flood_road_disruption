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

### Step 1: Preprocessing 
This step first calculates the centroids of all NUTS-2 and NUTS-3 regions. It then calculates for each country, the optimal routes between all NUTS-2 or NUTS-3 regions. The results of this step are saved in the preproc_output folder.<br />
Two import side-products of this step are: 
 - the scheduled nuts2/3_combinations.csv files, which determine the different number of AoIs that will be sampled in the next step; settings in this file may be overruled when the preprocessing is finished
 - a file showing the optimal (preferred) routes between the NUTS2/3-regions; these are a good quality check of the graphs and routing functionality
Inspect these files before going to the next step.<br /><br />

The preprocessing can be done by running the script: <i>preprocess_routes.py</i>

 



