"""
Created december 2020 - december 2021
@author: Frederique de Groen, Kees van Ginkel, Elco Koks
Partly based on the packages 'criticality tool' (Frederique) and 'TRAILS' (Elco)
=======
# -*- coding: utf-8 -*-
"""

import os, sys

sys.path.append(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\scripts")

import igraph as ig
from preprocess_routes import graph_load
import pandas as pd
import random
from statistics import mean
import time
import pygeos as pyg
from shapely import wkt
import numpy as np
from tqdm import tqdm
import feather
import pickle
import warnings

from utils import load_config

# translation between countrycodes (2- and 3-letter and country names)
config = load_config()
country_codes = config['paths']['data'] / 'country_codes.csv'
translate_cntr_codes = pd.read_csv(country_codes, delimiter=';').set_index('code3').to_dict(orient='dict')

# set paths
input_folder = r"D:\COACCH_paper\data" #TODO: change to config
#output_folder = r"P:\osm_flood\network_analysis\data\main_output\{}"
output_folder = config['paths']['main_output']

# parameters
AoI_name = 'AoI_RP100y_unique' #Todo: move these settings to config
weighing = 'time'  # time or distance #Todo: move these settings to config
#weighing = 'distance'

# import files
def import_graph(the_country, nuts_class='nuts3'):
    networks_europe_path = load_config()['paths']['graphs_folder']
    edge_file = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if
                f == the_country + '-edges.feather'][0]

    # read the network files from Elco Koks, the edges files already contain the flood hazard data
    network = pd.read_feather(edge_file)

    # create a geometry column with shapely geometries
    network['geoms'] = pyg.io.to_wkt(
        pyg.from_wkb(network.geometry))  # see if this should be activated with the new feather files
    network['geoms'] = network['geoms'].apply(wkt.loads)
    network.drop('geometry', axis=1, inplace=True)
    network.rename(columns={'geoms': 'geometry'}, inplace=True)
    # network = network.loc[~network['highway'].isin(['tertiary', 'tertiary_link'])]
    # todo: filter out the tertiary roads?

    # create the graph
    G = graph_load(network, ['geometry', 'id', 'RP100_cells_intersect', 'RP100_max_flood_depth',
                             'AoI_RP100y_majority', 'AoI_RP100y_unique', 'fds_majority', 'fds__unique'])

    # Add the nodes
    nodes = feather.read_dataframe(edge_file.replace("-edges", "-nodes_{}".format(nuts_class)))
    nodes.geometry = pyg.from_wkb(nodes.geometry)

    # Add the nodes to the graph
    G.add_vertices(len(nodes))
    G.vs['id'] = nodes['id']
    G.vs[nuts_class] = nodes[nuts_class]

    #print(G.summary())
    return G

from pathlib import Path

def import_optimal_routes(the_country):
    """
    Load the optimal routes between NUTS-X regions, as calculated during the preprocessing step
        *the_country* (string): Name of the country, should correspond to folder name in preproc_output

    Returns:
        *optimal_routes* (DataFrame) : dataframe with optimal_routes between NUTS-X regions
    """
    folder = Path(r'P:\osm_flood\network_analysis\data\preproc_output')
    folder = load_config()['paths']['preproc_output']
    file = folder / the_country / 'optimal_routes_{}_{}.feather'.format(weighing, the_country)
    optimal_routes = pd.read_feather(file)

    return optimal_routes

def aoi_combinations(all_aois_list, nr_comb, nr_iterations):
    return [random.choices(all_aois_list, k=nr_comb) for i in range(nr_iterations)]


def stochastic_network_analysis_phase1(G, nr_comb, nr_reps, country_code3, nuts_class, list_finished=None):
    """
    This function creates a folder structure with the experiments that are to be done in the percolation analysis,
    so that the actual experiments can be done using parallel processing

    Arguments:
        *G* (igraph Graph) : The network graph
        *nr_comb* (int) : The number of AoIs to remove
        *nr_reps* (in) : How often to repeat the AoI sampling, for the number of AoIs specified in nr_comb
        *country_code3 (string) : 3letter country code
        *nuts_class* (string) : can be 'nuts3' or 'nuts2'

    Effect:
        Creates a folder with the country name in 'main_output', with a subfolder 'scheduled',
        with subsubfolders with the nr of AoIs to sample at the same time (i.e. the nr_comb)
        and puts pickles in these folders
        These pickles contain several variables needed to carry out that experiment

    Returns: none
    """
    #Todo: if everything works well: remove the old os.path stuff
    current_country = translate_cntr_codes['country'][country_code3].lower()  # The country that is analysed
    # print("\nCurrent iteration is for:", current_country)

    output_folder = load_config()['paths']['main_output']
    assert output_folder.exists()
    newpath = output_folder / current_country / 'scheduled' / str(nr_comb)
    if not newpath.exists(): newpath.mkdir(parents=True)

    all_aois = list(set([item for sublist in G.es[AoI_name] for item in sublist if item != 0 and item == item]))

    # for only 1 AoI and the maximum nr of AoI's, there is a slightly different approach
    if nr_comb == 1:
        list_aois = all_aois
    elif nr_comb == len(all_aois):
        list_aois = [all_aois]
    else:
        if list_finished is not None:
            print("this function should be found in one of the scripts of COACCH but not used now")
            # list_aois = aoi_combinations_except(all_aois, nr_comb, nr_reps, list_finished)
        else:
            list_aois = aoi_combinations(all_aois, nr_comb, nr_reps)

    for i, aoi in enumerate(list_aois):
        # i indicates the index of the experiments
        # each experiment is a unique combination of AoI's disrupted at the same time
        filename = output_folder / current_country / 'scheduled' / str(nr_comb) / (str(i) + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump((nr_comb, aoi, i, current_country, country_code3, nuts_class), f)


def stochastic_network_analysis_phase2(tup):
    """
    #todo: update documentation
    Input argument:
        *tup* (tuple) = tuple of lenght 6, containing the the nr_comb and the unique i (ID) of the experiment
    """
    special_setting = None #'depth_threshold'
    depth_threshold = 0. #depth threshold in m.
    if special_setting is not None:
        extension = ""
        if special_setting == 'depth_threshold':
            extension = extension + "Threshold: " + str(depth_threshold) + " m"
        warnings.warn("Stochatistic_network_analysis_phase2() in percolation_optimized_parallel.py " +
                      "runs with special setting {}, {}".format(special_setting,extension))


    output_folder = load_config()['paths']['main_output']

    # read tuple
    nr_comb, aoi, i, country_name, country_code3, nutsClass = tup[0], tup[1], tup[2], tup[3], tup[4], tup[5]

    # SKIP IF NR of COMBINATIONS OF AOI IS ALREADY FINISHED BY CHECKING IF OUTPUT FILE IS ALREADY CREATED
    #result_path = os.path.join(output_folder.format(country_name), 'finished', str(nr_comb))
    result_path = output_folder / country_name / 'finished' / str(nr_comb)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.exists(os.path.join(result_path,str(i) + '.csv')):
        print('nr_comb = {}, experiment = {} already finished!'.format(nr_comb,i))
        return None

    else:
        # do the calculations
        # Import graph and OD matrix of optimal routes
        G = import_graph(country_code3, nuts_class=nutsClass)
        od_optimal_routes = import_optimal_routes(country_name)

        # initiate variables
        df = pd.DataFrame(columns=['AoI combinations', 'disrupted', 'avg extra time', 'AoI removed', 'no detour'])
        tot_routes = len(od_optimal_routes.index)

        start = time.time()

        # remove the edges per flood event (Area of Influence)
        if nr_comb == 1:
            to_remove = G.es.select(lambda e: aoi in e.attributes()[AoI_name])
        else:
            to_remove = G.es.select(lambda e: set(aoi) & set(e.attributes()[AoI_name]))

        #Iterate over edges that are planned to be removed based on their AoI info, to do further filtering
        if special_setting == 'depth_threshold':
            #only remove edges that are inundated above a certain threshold
            to_remove = [edge for edge in to_remove if edge['RP100_max_flood_depth'] >= depth_threshold]

        G.delete_edges(to_remove)

        extra_time = []
        disrupted = 0
        nr_no_detour = 0
        for ii in range(len(od_optimal_routes.index)):
            o, d = od_optimal_routes.iloc[ii][['o_node', 'd_node']]

            # calculate the (alternative) distance between two nodes
            # now the graph is treated as directed, make mode=ig.ALL to treat as undirected
            ### Todo: this can be optimized. You only need to recalculate routes which are disrupted!!!!
            ### i.e. only calculate route if any edge in the route between OD-pair is in to_remove
            ### todo: it would be good to save the names of the OD-pairs that are disrupted
            alt_route = G.shortest_paths_dijkstra(source=int(o), target=int(d), mode=ig.OUT, weights=weighing)
            alt_route = alt_route[0][0]
            if alt_route != np.inf:
                # alt_route = inf if the route is not available
                # append to list of alternative routes to get the average
                extra_time.append(alt_route - od_optimal_routes.iloc[ii][weighing]) #changed 'time' into weighing
                if od_optimal_routes.iloc[ii][weighing] != alt_route: #changed 'time' into weighing
                    # the alternative route is different from the preferred route
                    disrupted += 1
            else:
                # append to calculation dataframe
                disrupted += 1
                nr_no_detour += 1

        df = df.append({'AoI combinations': nr_comb, 'disrupted': disrupted / tot_routes * 100,
                        'avg extra time': mean(extra_time), 'AoI removed': aoi, 'no detour': nr_no_detour / tot_routes * 100},
                       ignore_index=True)
        df.to_csv(os.path.join(result_path, str(i) + '.csv'),sep=';')

    end = time.time()
    print('Finished percolation subprocess. Nr combinations: {}, Experiment nr: {}, time elapsed: {}'.format(nr_comb, i, end - start))

