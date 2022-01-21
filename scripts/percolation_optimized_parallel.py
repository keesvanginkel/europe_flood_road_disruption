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
from pathlib import Path
import pygeos as pyg
from shapely import wkt
import numpy as np
from tqdm import tqdm
import feather
import pickle
import warnings
import json

from utils import load_config

import logging

# translation between countrycodes (2- and 3-letter and country names)
#Todo: these config loads should be avoided; directly load from europe_utils
config = load_config(file='config.json')
country_codes = config['paths']['data'] / 'country_codes.csv'
warnings.warn("""Still need to fix issue with loading country codes here""")
translate_cntr_codes = pd.read_csv(country_codes, delimiter=';').set_index('code3').to_dict(orient='dict')

# set paths
#input_folder = r"D:\COACCH_paper\data" #TODO: change to config
#output_folder = r"P:\osm_flood\network_analysis\data\main_output\{}"
#output_folder = config['paths']['main_output']

# parameters
AoI_name = 'AoI_RP100y_unique' #Todo: move these settings to config
weighing = 'time'  # time or distance #Todo: move these settings to config
#weighing = 'distance'

# import files
def import_graph(the_country, nuts_class='nuts3',config_file='config.json'):
    """
    Arguments:
        *the_country* (string) : 3-letter code of country name e.g. 'Bel'
        *nuts_class* (string) : 'nuts3' or 'nut2'
        *config_file* (string) : name of the config file directing to the path, default = config.json
    
    Returns:
        **
    
    """
    config = load_config(file=config_file)
    networks_europe_path = config['paths']['graphs_folder']
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
    # network = network.loc[~network['highway'].isin(['tertiary', 'tertiary_link'])] #to filter out, not recommended
    # ... because it could result in many nodes of degree 1 or even 0. Better to cleanup graph first.

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



def import_optimal_routes(the_country,config_file='config.json'):
    """
    Load the optimal routes between NUTS-X regions, as calculated during the preprocessing step
        *the_country* (string): Name of the country, should correspond to folder name in preproc_output

    Returns:
        *optimal_routes* (DataFrame) : dataframe with optimal_routes between NUTS-X regions
    """
    config = load_config(file=config_file)
    folder = config['paths']['preproc_output']
    file = folder / the_country / 'optimal_routes_{}_{}.feather'.format(weighing, the_country)
    optimal_routes = pd.read_feather(file)

    return optimal_routes

#def aoi_combinations(all_aois_list, nr_comb, nr_iterations):
#    ### THIS FUNCTION CONTAINS AN ERROR, THIS DOES NOT RETURN COMBINATIONS!
#    return [random.choices(all_aois_list, k=nr_comb) for i in range(nr_iterations)]

def aoi_combinations(all_aois_list, nr_comb, nr_iterations):
    """"
    Draws multiple combinations from a (larger) set of AoIs

    Arguments:
        *all_aois_list* (list) : list containing all the possible aois to draw from
        *nr_comb* (int) : how much items one selection of the set must contain
        *nr_iterations* (int) : how often a selection from the set is to be made

    Two improvements over version 1:
         - Draw REAL combinations per selection, without repetition. In vs 1, repetion of aoi in one selection was possible
         - Avoid duplicate selections
    """
    assert len(all_aois_list) == len(set(all_aois_list)) ### make sure there are no duplicates in the aoi list

    from itertools import combinations
    from random import shuffle
    #Check that there are enough combinations
    from math import comb

    ncr =  comb(len(all_aois_list),nr_comb)
    if ncr <= nr_iterations:
        warnings.warn("Can only draw {} combinations from {} aois, while {} were requested".format(
                ncr,len(all_aois_list),nr_iterations))
        nr_iterations = ncr #Limit the nr of subsequences to the maximum theoretical possible #Todo check if our approach always finds these

    result = []
    # Because the itertools generator draws combinations in sorted order, for safety we shuffle list each time
    while len(result) < nr_iterations:
        shuffle(all_aois_list)
        Iterator = combinations(all_aois_list,nr_comb)
        subsequence = next(Iterator)
        assert len(subsequence) == len(set(subsequence)) #check for duplicates within one subsequence
        set_subsequence  = set(subsequence) #first save as sets to easy check for duplicates
        if set_subsequence not in result: #check for duplicates between subsequences
            result.append(set_subsequence)

    result = [list(r) for r in result] #convert sets to lists

    return result


def stochastic_network_analysis_phase1(config_file,G, nr_comb, nr_reps, country_code3, nuts_class, list_finished=None):
    """
    This function creates a folder structure with the experiments that are to be done in the percolation analysis,
    so that the actual experiments can be done using parallel processing

    Arguments:
        *config_file* (str) : name of the config file, to be given to load_config from utils.py
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
    current_country = translate_cntr_codes['country'][country_code3].lower()  # The country that is analysed
    # print("\nCurrent iteration is for:", current_country)

    config = load_config(file=config_file)
    output_folder = config['paths']['main_output']
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
        list_aois = aoi_combinations(all_aois, nr_comb, nr_reps)

    for i, aoi in enumerate(list_aois):
        # i indicates the index of the experiments
        # each experiment is a unique combination of AoI's disrupted at the same time
        filename = output_folder / current_country / 'scheduled' / str(nr_comb) / (str(i) + '.pkl')

        #Todo? We possibly could save the scheduled aois as csv so that they are more easy to inspect.
        with open(filename, 'wb') as f:
            pickle.dump((nr_comb, aoi, i, current_country, country_code3, nuts_class), f)

def stochastic_network_analysis_phase1_event_sampling(config_file,country_code3, nuts_class, year, data,
                                                      special_setting='todo'):
    """
    This function creates a folder structure with the experiments that are to be done in the percolation analysis,
    so that the actual experiments can be done using parallel processing, for event-based dataset

    Arguments:
        *config_file* (str) : name of the config file, to be given to load_config from utils.py
        *country_code3 (string) : 3letter country code
        *nuts_class* (string) : can be 'nuts3' or 'nuts2'

    Effect:
        Creates a folder with the country name in 'main_output', with a subfolder 'scheduled_event',
        and puts pickles in these folders
        These pickles contain several variables needed to carry out that experiment

    Returns: none
    """
    current_country = translate_cntr_codes['country'][country_code3].lower()  # The country that is analysed
    # print("\nCurrent iteration is for:", current_country)

    config = load_config(file=config_file)
    output_folder = config['paths']['main_output']
    assert output_folder.exists()



    warnings.warn("""Note that we remapped some variables to be able 
                    to stil use the stochatistic_network_analysis_phase2()_function: 
                    nr_comb : 'year' of the event 
                    i : 'refers to sampling variant, '0: means default variant' (still unused)
                    """)

    nr_comb = year
    i = 0
    aoi = data['c_aoi']

    newpath = output_folder / current_country / 'scheduled' / str(nr_comb)
    if not newpath.exists(): newpath.mkdir(parents=True)

    filename = output_folder / current_country / 'scheduled' / str(nr_comb) / (str(i) + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump((nr_comb, aoi, i, current_country, country_code3, nuts_class), f)

def make_pool_logger_phase2(nr_comb,i):
    """
    Create a logger for phase 2, that works with parallel processing.

    """
    #todo: add aoi and nr in log name
    nr_comb = str(nr_comb)
    i = str(i)
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(funcName)20s()] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger("{}-{}-{}".format(__name__,nr_comb,i))
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    #logger.info('Logger is created')
    return logger

def common_member(a, b):
    '''
    Check lists for common members
    from: https://www.geeksforgeeks.org/python-check-two-lists-least-one-element-common/

    '''
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True)
    return(False)

def stochastic_network_analysis_phase2(tup):
    """
    Carry out the percolation analysis which has been scheduled in phase 1.

    Arguments:
        *tup* (tuple) = tuple of length 11, containing:
            (config_file, nr_comb, aoi, i, country_name, country_code3, nuts_level, special_setting,G,cehck_es,route_algorithm_)
                config_file (str)            : name of the config file, to be given to load_config from utils.py
                nr_comb (int)               : number of combinations, i.e. microfloods sampled at the same time
                aoi (int or list of int)    : the microfloods sampled
                i (int)                     : identifier of the experiment (corresponds to scheduled/finished filename)
                country_name (string)       : full lowercase country name (e.g. 'belgium')
                country_code3 (str)         : 3-letter countryname (e.g. 'BEL')
                nuts_level (str)            : can be 'nuts3' or 'nuts2'
                special_setting (str)       : run analysis in non-default mode, used for uncertainty analysis
                    Available special settings:
                        'giant_component'   : analyse the size of the giant component (largest connected cluster)
                        'depth_threshold'   : impose water depth threshold on the removal of edges
                    Other special settings are already imposed in the preprocessing step
                G (igraph Graph object)     : the undisturbed network graph
                check_es                : number of edges in the undisturbed graph (used as checksum)
                route_algorithm (string)    : can be 'version_1' or 'version_2' or 'version_3'
    Returns: None

    Effect:
        Write results of the percolation analysis to the 'main_output/*name*/finished' path set in the config file

    Known issues: when calling the function iteratively, sometimes the manipulations are not done on the original graph,
    but on the output of the previous function call. Therefore, there is a check on the number of edges
    """

    start = time.time()

    #update version > 1.0
    assert len(tup) == 11

    # unpack tuple
    config_file, nr_comb, aoi, i, country_name, country_code3, nutsClass, special_setting, G, check_es, route_algorithm_ = \
        tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6], tup[7], tup[8], tup[9],tup[10]

    # INSTANTIATE LOGGING FUNCTIONALITY
    logger = make_pool_logger_phase2(nr_comb,i)
    logger.info('Logging {} nr_comb = {}, experiment = {}'.format(country_code3,nr_comb, i))

    config = load_config(file=config_file)

    try:
        #special_setting = 'giant_component' #'depth_threshold', 'giant_component'
        #depth_threshold = 0. #depth threshold in m.
        if special_setting is not None:
            extension = ""
            if special_setting == 'depth_threshold':
                extension = extension + "Threshold: " + str(depth_threshold) + " m"
            warnings.warn("Stochatistic_network_analysis_phase2() in percolation_optimized_parallel.py " +
                          "runs with special setting {}, {}".format(special_setting,extension))

        output_folder = config['paths']['main_output']

        # SKIP IF NR of COMBINATIONS OF AOI IS ALREADY FINISHED BY CHECKING IF OUTPUT FILE IS ALREADY CREATED
        result_path = output_folder / country_name / 'finished' / str(nr_comb)
        result_path.mkdir(exist_ok=True,parents=True)

        result_file = result_path / (str(i) + '.csv')
        if result_file.exists():
            logger.info('nr_comb = {}, experiment = {} already finished!'.format(nr_comb,i))
            return None

        #only do the calculations if all above checks are met
        else:
            check_n_es = len(G.es)
            check_n_vs = len(G.vs)
            logger.info('Worker before percolation: nr edges|vertices original graph: {} | {}'.format(check_n_es, check_n_vs))

            if not check_n_es == check_es:
                raise Exception("""The number of edges of the input graph: {} does not match with the original graph: {}.
                                This might be due to iteratively calling this function.""".format(check_n_es,check_n_vs))

                        # Import graph and OD matrix of optimal routes
            #G = import_graph(country_code3, nuts_class=nutsClass,config_file=config_file)
            od_optimal_routes = import_optimal_routes(country_name,config_file=config_file)


            t1 = time.time()
            logger.info('start - t1: {} sec passed till importing optimal routes check'.format(t1 - start))

            # initiate variables
            ser = pd.Series(name=i,
                            index=['AoI combinations',
                                   'experiment',
                                   'disrupted',
                                   'no detour',
                                   'avg extra time',
                                   'AoI removed',
                                   'OD-disrupted',
                                   'OD-with_detour',
                                   'with_detour_extra_times'],
                            dtype=object)
            tot_routes = len(od_optimal_routes.index)



            # remove the edges per flood event (Area of Influence)
            if nr_comb == 1:
                to_remove = G.es.select(lambda e: aoi in e.attributes()[AoI_name])
            else:
                to_remove = G.es.select(lambda e: set(aoi) & set(e.attributes()[AoI_name]))

            # Iterate over edges that are planned to be removed based on their AoI info, to do further filtering
            if special_setting == 'depth_threshold':
                # only remove edges that are inundated above a certain threshold
                to_remove = [edge for edge in to_remove if edge['RP100_max_flood_depth'] >= depth_threshold]

            # identify the od_optimal_routes that are affected, i.e. they have at least one edge that is to be removed
            aff = od_optimal_routes['e_ids'].apply(lambda x: common_member(json.loads(x), to_remove.indices))
            conc = od_optimal_routes[aff]['origin'] + '-' + od_optimal_routes[aff]['destination']
            affected_OD_pairs = list(conc.values)



            t2 = time.time()
            logger.info('t1-t2: {} sec passed for selection to removed edges'.format(t2 - t1))





            if special_setting == 'giant_component':
                result_gc_path = output_folder / country_name / 'finished_gc'
                if not result_gc_path.exists():
                    result_gc_path.mkdir(parents=False, exist_ok=True)

                # Calculate reference metrics for undisturbed graph, only in the first iteration
                file = result_gc_path / 'reference.csv'
                if not file.exists():
                    logger.info('calculating refence metrics for giant component analysis, saving to {}'.format(file))
                    edges_in_graph, nodes_in_graph, edges_in_giant, nodes_in_giant = giant_component_analysis(G,mode='strong')
                    d = {'ref_edges_in_graph' : edges_in_graph,
                         #'ref_nodes_in_graph' : nodes_in_graph, #does not change because we do edge percolation
                         'ref_edges_in_giant' : edges_in_giant} #,
                         #'ref_nodes_in_giant' : nodes_in_giant}
                    result = pd.DataFrame(pd.Series(data=d,name='undisturbed graph')).T
                    result.to_csv(file,sep=';')

            G.delete_edges(to_remove)

            t3 = time.time()
            logger.info('t2-t3: {} sec passed after deleting edges'.format(t3 - t2))

            x = od_optimal_routes[aff][['o_node', 'd_node']]

            ### FIRST APPROACH TO ROUTE CALCULATION (USED IN THE VERSION 1.0 RELEASE) ###

            assert len(affected_OD_pairs) == aff.sum()
            logger.info('{} affected routes'.format(len(affected_OD_pairs)))

            if route_algorithm_ == 'version_1':
                new_routes = calculate_shortest_paths_dijkstra(G, [int(v) for v in x['o_node']],
                                                            [int(v) for v in x['d_node']])

                old_times = od_optimal_routes[aff]['time']
                new_times = pd.Series(index=old_times.index,data=new_routes)

                ###########################################
                t4 = time.time()
                logger.info('t3-t4: {} sec passed for calculating new routes, used algorithm 1'.format(t4 - t3))

            elif route_algorithm_ == 'version_2':
            ### SECOND APPROACH TO ROUTE CALCULATION  ###
                y_sel = calculate_shortest_paths_matrix(G, x,weighing=weighing)
                #Todo: weird variable name 'new routes', confusing compared to implementation of other algs.
                new_routes = od_optimal_routes[aff][['o_node', 'd_node', 'origin', 'destination', 'time']]
                d = y_sel.unstack(-1).loc[[(row,col) for col, row in zip(new_routes.o_node, new_routes.d_node)]].reset_index()
                #new_routes['new_time'] = None
                # Make sure that the lists are not shuffled
                assert list(zip(d.o_node, d.d_node)) == list(zip(new_routes.o_node, new_routes.d_node))
                #new_routes['new_time'] = d[0].values

                old_times = od_optimal_routes[aff]['time']
                new_times = pd.Series(index=old_times.index, data=d[0].values)
                assert len(old_times) == len(new_times)

                t4 = time.time()
                logger.info('t3-t4: {} sec passed for calculating new routes, used  algorithm 2'.format(t4 - t3))


            ### THIRD APPROACH TO ROUTE CALCULATION  ###
            elif route_algorithm_ == 'version_3':
                check_duplicates(x) #A duplicate if: OD-pair also exists as DO-pair
                #so far, there seem to be no duplicates in x, so no need to handle these
                new_routes = calculate_shortest_paths_speedup(G, [int(v) for v in x['o_node']], [int(v) for v in x['d_node']])

                old_times = od_optimal_routes[aff]['time']
                new_times = pd.Series(index=old_times.index, data=new_routes)
                t4 = time.time()
                logger.info('t3-t4: {} sec passed for calculating new routes, used  algorithm 3'.format(t4 - t3))

            #### Do some general processing of the results to save in usable output metrics ####

            # Split detours from no detours
            no_detours_mask = new_times == float('inf')  # note, this is a bool mask of only the disrupted routes...
            od_optimal_routes['no_detours'] = no_detours_mask[no_detours_mask]  # ...while this is the mask of all
            od_optimal_routes['no_detours'] = od_optimal_routes['no_detours'].fillna(False)

            with_detours_mask = ~no_detours_mask
            od_optimal_routes['with_detours'] = with_detours_mask[
                with_detours_mask]  # ...while this is the mask of all
            od_optimal_routes['with_detours'] = od_optimal_routes['with_detours'].fillna(False)
            assert aff.sum() == od_optimal_routes['with_detours'].sum() + od_optimal_routes['no_detours'].sum()

            OD_with_detour = list(od_optimal_routes[od_optimal_routes['with_detours']]['origin'] + '-' + \
                                  od_optimal_routes[od_optimal_routes['with_detours']]['destination'])

            extra_time_per_route = new_times - old_times

            ### Handle function output
            ser.at['AoI combinations'] = nr_comb
            ser.at['experiment'] = ser.name
            ser.at['disrupted'] = len(affected_OD_pairs) / tot_routes * 100
            if len(extra_time_per_route) == 0:
                ser.at['avg extra time'] = 0
            else:
                ser.at['avg extra time'] = (extra_time_per_route[~no_detours_mask]).mean()
            ser.at['AoI removed'] = json.dumps(np.array(aoi).tolist())
            ser.at['no detour'] = no_detours_mask.sum() / tot_routes * 100
            ser.at['OD-disrupted'] = json.dumps(affected_OD_pairs)
            ser.at['OD-with_detour'] = json.dumps(OD_with_detour)
            ser.at['with_detour_extra_times'] = json.dumps(
                ['{:.3f}'.format(t) for t in extra_time_per_route[~no_detours_mask]])


            if special_setting == 'giant_component':
                gc_start = time.time()
                # Calculate the metrics for the Giant Component analysis
                edges_in_graph, nodes_in_graph, edges_in_giant, nodes_in_giant = giant_component_analysis(G,
                                                            mode='strong')
                d = {'edges_in_graph': edges_in_graph,
                     #'nodes_in_graph': nodes_in_graph, #since we do edge percolation, no need to save these
                     'edges_in_giant': edges_in_giant
                     #'nodes_in_giant': nodes_in_giant
                     }
                ser = ser.append(pd.Series(d))
                gc_end = time.time()
                logger.info('gc_start - gc end: {} sec for counting giant component edges'.format(gc_end - gc_start))

            # this gives a transposed output compared to version 1, not very handy for backward compat.
            # ser.to_csv((result_path / '{}.csv'.format(i)),sep=';',index=True,header=False)
            pd.DataFrame(ser).T.to_csv((result_path / '{}.csv'.format(i)), sep=';', header=True, index=False)



        end = time.time()
        logger.info('t4-tend: {} sec passed up till saving the routes'.format(end - t4))

        #check_n_es = len(G.es)
        #check_n_vs = len(G.vs)
        #logger.info(
        #    'Worker after percolation: nr edges|vertices percolated graph: {} | {}'.format(check_n_es, check_n_vs))

        logger.info('Finished percolation subprocess. Nr combinations: {}, Experiment nr: {}, time elapsed: {}'.format(nr_comb, i, end - start))

    except Exception as Argument:
        #only save log if exceptions arise
        logger_filename = config['paths']['logs'] / 'log_{}_{}_{}_{}_{}.log'.format(
                    country_name, nutsClass, 'pool_phase2',nr_comb,i,)

        fileHandler = logging.FileHandler(logger_filename)
        logFormatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(funcName)20s()] [%(levelname)-5.5s]  %(message)s")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        logger.exception(str(Argument))

def check_duplicates(OD_df):
    """
    Check for duplicates in OD-dataframe and raises a warning when duplicates exist

    Arguments:
        *OD_df* (DataFrame) : Mandatory columns are ['o_node'] and ['d_node'], each row is a route
    """
    OD = list(zip(OD_df['o_node'], OD_df['d_node']))
    DO = list(zip(OD_df['d_node'], OD_df['o_node']))
    if (len(OD) + len (DO)) != len(list(set(OD + DO))):
        warnings.warn("There are duplicates in the OD route list, i.e. at least one A->B exists as B->A")




def calculate_shortest_paths_speedup(G, o_nodes, d_nodes, weighing='time'):
    """Fast way to calculate the shortest paths for a set of routes in a undirected graph
    Note that list(zip(o_nodes,d_nodes)) should return the list of OD-pairs
    #todo: unique?

    Arguments:
        *G* (igraph Graph)  : the (undirected) graph
        *o_nodes* (list)    : list of origin vertices (ints)
        *d_nodes* (list)    : list of destinatination vertices (ints)
        *weighing* (string) : weighing object in the graph

    Returns:
        path_times  (list)     : length (could also be time) of the routes; len(path_times) == len(o_nodes)
    """
    # optional: a speedup by providing multiple destinations at the same time where possible...
    path_times = []

    for OD_pair in list(zip(o_nodes, d_nodes)):
        paths = G.get_shortest_paths(OD_pair[0], OD_pair[1], weights=weighing,
                                     output='epath')
        path = paths[0]
        if len(path) == 0:
            path_time = float('inf') # was np.NaN in Elco's version
        else:
            path_time = sum(G.es[path]['time'])
        path_times.append(path_time)

    return path_times


def calculate_shortest_paths_matrix(G, x, weighing='time'):
    """Original way to calculate the shortest paths for a set of routes in a undirected graph in release 1.0

    Note that list(zip(o_nodes,d_nodes)) should return the list of OD-pairs
    #todo: unique?

    Arguments:
        *G* (igraph Graph)  : the (undirected) graph
        *x* (DataFrame) : should contain columns x['o_node'] and x['d_node']
        *weighing* (string) : weighing object in the graph

    Returns:
        *y_sel* (DataFrame) : matrix with the travel times
    """
    # Make unique OD-matrix for the affected routes
    # peculiarity: will calculate routes between all affected origins AND all affected destinations
    # note that these are more routes than the unique OD-pairs that were effected; therefore we need a pivot
    # Create pivot table of the affected routes
    x['values'] = True
    bool_aff_OD_pivot = x.pivot(index='o_node', columns='d_node', values='values').fillna(value=False)

    aff_origins = [int(v) for v in bool_aff_OD_pivot.index]
    aff_destinations = [int(v) for v in bool_aff_OD_pivot.columns]
    shortest_paths = G.shortest_paths(source=aff_origins, target=aff_destinations, weights=weighing)
    y = pd.DataFrame(shortest_paths, index=bool_aff_OD_pivot.index, columns=bool_aff_OD_pivot.columns)
    y_sel = y[bool_aff_OD_pivot]

    return y_sel

def calculate_shortest_paths_dijkstra(G, o_nodes, d_nodes, weighing='time'):
    """Original way to calculate the shortest paths for a set of routes in a undirected graph in release 1.0

    Note that list(zip(o_nodes,d_nodes)) should return the list of OD-pairs
    #todo: unique?

    Arguments:
        *G* (igraph Graph)  : the (undirected) graph
        *o_nodes* (list)    : list of origin vertices (ints)
        *d_nodes* (list)    : list of destinatination vertices (ints)
        *weighing* (string) : weighing object in the graph

    Returns:
        path_times  (list)     : length (could also be time) of the routes; len(path_times) == len(o_nodes)
    """
    # optional: a speedup by providing multiple destinations at the same time where possible...
    path_times = []

    for OD_pair in list(zip(o_nodes, d_nodes)):
        path_time = G.shortest_paths_dijkstra(source=OD_pair[0], target=OD_pair[1], weights=weighing,
                                              mode=ig.OUT)[0][0]
        path_times.append(path_time)

    return path_times

def giant_component_analysis(G,mode='strong'):
    """
    Calculate the number of edges in the largest connected component of the graph.
    Within the function, the giant is called 'H' (an igraph Graph)

    Arguments:
        *G* (igraph Graph) : iGraph graph object
        *mode* (string) : mode for clustering G ('strong' or 'weak'),
            say that the the new network is clustered in two parts, a small part S and a large part H (giant)
            'strong' means that within each cluster you can travel from all
                          see https://igraph.org/python/doc/api/igraph.Graph.html#clusters
                          for more info see: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)
        Note that for undirected graphs (U---) there is no difference...

    Returns:
        *edges_in_graph* (int) : number of edges in the total (disrupted) graph
        *nodes_in_graph* (int) : number of nodes in the total (disrupted) graph
        *edges_in_giant* (int) : number of edges in the giant component of the graph
        *nodes_in_giant* (int) : number of nodes in the giant component of the graph
    """
    edges_in_graph = G.ecount()
    nodes_in_graph = G.vcount()
    giant = G.clusters(mode='strong').giant()
    edges_in_giant = giant.ecount()
    nodes_in_giant = G.ecount()

    return edges_in_graph, nodes_in_graph, edges_in_giant, nodes_in_giant


