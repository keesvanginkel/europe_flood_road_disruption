# Europe_flood_road_disruption
This repo contains python code to do percolation analysis for road networks of 30 European countries. The results are described in the paper <i>Will river floods tip European road networks?</i>, by Kees van Ginkel, Elco Koks, Frederique de Groen, Viet Dung Nguyen and Lorenzo Alfieri (2021, under review).

In summary, the model works as follows. The road network of each country is fetched from [OpenStreetMap](www.openstreetmap.org). Then, the centroids of the Eurostat-2016 [NUTS-2 or NUTS-3 regions](https://ec.europa.eu/eurostat/web/nuts/background) within the country are determined, to calculate the preferred routes (shortest travel time) between all NUTS-regions the country. Next, the road network is overlayed with the 100x100 m River flood hazard from the [Joint Research Centre LISFLOOD-FP model](https://data.jrc.ec.europa.eu/dataset/85470f72-9406-4a91-9f1f-2a0220a5fa86), as described in [Dottori et al, (2021, under review)](https://essd.copernicus.org/preprints/essd-2020-313/essd-2020-313.pdf). Then, the percolation analysis begins. Each time, a synthetic flood event composing of one or multiple microfloods (Areas of Influence, originating from a 5x5 km grid cell) hits some roads in the network of the country, which are temporarily removed from the netwerk graph. For this disruption, the routes are recaculcated, and the differences with the undisturbed situation are measured with three Metrics. Having repeated this for many events with increasing magnitude, one can infer how the road network performance deteriorates from increasingly large floods.

# Code overview
Note that the paths are set in the config.json file, which is loaded in utils.py.<br />
Besides these generic utils, there are some specific functions for handling data on the EU (NUTS) classification, in Europe_utils.py

### Step 0: Preprocessing with TRAILS (outside this Github)
This step is done outside this Github project. First, the python package [TRAILS](https://github.com/BenDickens/trails) is used to prepare clean graphs ([python iGraph objects](https://igraph.org/python/) from an OpenStreetMap planet dump. This is done for individual countries. The network of each country is saved as an edges.feather (vertices) and nodes.feather file. This file format can be easily read with (Geo)Pandas. <br />
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

