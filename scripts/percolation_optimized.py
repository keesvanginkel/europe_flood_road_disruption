# -*- coding: utf-8 -*-
"""
Created on 2-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import os, sys

sys.path.append(r"D:\COACCH_paper\europe_flood_road_disruption\scripts")

# import networkx as nx
import igraph as ig
from preprocess_routes import graph_load
import pandas as pd
import random
from statistics import mean
import time
import pygeos as pyg
from shapely import wkt
import copy
import numpy as np
from tqdm import tqdm
import feather

# translation between countrycodes (2- and 3-letter and country names)
translate_cntr_codes = pd.read_csv(r"D:\COACCH_paper\europe_flood_road_disruption\data\country_codes.csv",
                                delimiter=';').set_index('code3').to_dict(orient='dict')

# set paths
country = 'BEL'  # Albania
input_folder = r"D:\COACCH_paper\data"
current_country = translate_cntr_codes['country'][country].lower()  # The country that is analysed
print("\nCurrent iteration is for:", current_country)
output_folder = r"D:\COACCH_paper\data\output\{}".format(current_country)

# parameters
nr_reps = 10
AoI_name = 'AoI_RP100y_unique'
weighing = 'time'  # time or distance

# import files
# G_nx = nx.read_gpickle(r"P:\osm_flood\network_analysis\networkx\{c}\{c}_graph.gpickle".format(c=country))  # NETWORKX
# pref_time = gpd.read_file(r"P:\osm_flood\network_analysis\networkx\{c}\time_pref_routes.shp".format(c=country))  # NETWORKX
networks_europe_path = os.path.join(input_folder, 'networks_intersect_hazard_elco_koks')
edge_file = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if
            f == country + '-edges.feather'][0]

# read the network files from Elco Koks
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
nuts_class = 'nuts2'
nodes = feather.read_dataframe(edge_file.replace("-edges", "-nodes_{}".format(nuts_class)))
nodes.geometry = pyg.from_wkb(nodes.geometry)

# Add the nodes to the graph
G.add_vertices(len(nodes))
G.vs['id'] = nodes['id']
G.vs[nuts_class] = nodes[nuts_class]

print(G.summary())

od_optimal_routes = pd.read_feather(os.path.join(output_folder.format(current_country),
                 'optimal_routes_{}_{}.feather'.format(weighing, current_country)))


# # convert NetworkX graph to iGraph (identifiers change)  # NETWORKX
# G = ig.Graph.from_networkx(G_nx)
# print(ig.summary(G))
#
# # Create a dictionary to convert between the networkx id and igraph id.
# # Warning! The ids of igraph are updated when e.g. edges are removed.
# vid_lookup = {}
# for i in range(len(G.vs)):
#     vid_lookup[G.vs()[i]['_nx_name']] = i
# print(vid_lookup)  # NETWORKX


def aoi_combinations(all_aois_list, nr_comb, nr_iterations):
    return [random.choices(all_aois_list, k=nr_comb) for i in range(nr_iterations)]


def stochastic_network_analysis(nr_comb):
    # SKIP IF NR of COMBINATIONS OF AOI IS ALREADY FINISHED BY CHECKING IF OUTPUT FILE IS ALREADY CREATED
    if os.path.exists(os.path.join(output_folder, 'aoi_{}.csv'.format(nr_comb))):
        print('{} already finished!'.format(nr_comb))
        return None

    all_aois = list(set([item for sublist in G.es[AoI_name] for item in sublist if item != 0 and item == item]))

    # for only 1 AoI and the maximum nr of AoI's, there is a slightly different approach
    if nr_comb == 1:
        list_aois = all_aois
    elif nr_comb == len(all_aois):
        list_aois = [all_aois]
    else:
        list_aois = aoi_combinations(all_aois, nr_comb, nr_reps)

    # initiate variables
    df = pd.DataFrame(columns=['AoI combinations', 'disrupted', 'avg extra time', 'AoI removed', 'no detour'])
    tot_routes = len(od_optimal_routes.index)

    start = time.time()
    for i, aoi in tqdm(enumerate(list_aois), desc='Iterating over AoIs'):
        # Make a copy of the graph
        H = copy.deepcopy(G)

        # remove the edges per flood event (Area of Influence)
        if nr_comb == 1:
            to_remove = G.es.select(lambda e: aoi in e.attributes()[AoI_name])
        else:
            to_remove = G.es.select(lambda e: set(aoi) & set(e.attributes()[AoI_name]))
        H.delete_edges(to_remove)

        extra_time = []
        disrupted = 0
        nr_no_detour = 0
        for ii in range(len(od_optimal_routes.index)):
            o, d = od_optimal_routes.iloc[ii][['o_node', 'd_node']]

            # o = vid_lookup[int(o)]  # NETWORKX
            # d = vid_lookup[int(d)]  # NETWORKX

            # calculate the (alternative) distance between two nodes
            # now the graph is treated as directed, make mode=ig.ALL to treat as undirected
            alt_route = H.shortest_paths_dijkstra(source=int(o), target=int(d), mode=ig.OUT, weights=weighing)
            alt_route = alt_route[0][0]
            if alt_route != np.inf:
                # alt_route = inf if the route is not available
                # append to list of alternative routes to get the average
                extra_time.append(alt_route - od_optimal_routes.iloc[ii]['time'])
                if od_optimal_routes.iloc[ii]['time'] != alt_route:
                    # the alternative route is different from the preferred route
                    disrupted += 1
            else:
                # append to calculation dataframe
                disrupted += 1
                nr_no_detour += 1

        df = df.append({'AoI combinations': nr_comb, 'disrupted': disrupted / tot_routes * 100,
                        'avg extra time': mean(extra_time), 'AoI removed': aoi, 'no detour': nr_no_detour / tot_routes * 100},
                       ignore_index=True)

    df.to_csv(os.path.join(output_folder, 'aoi_{}.csv'.format(nr_comb)))
    end = time.time()
    print('Nr combinations: {}, time elapsed: {}'.format(nr_comb, end - start))


if __name__ == '__main__':
    nr_combinations = [1,2,3,4,5,10,20,30,40,50]
    for comb in nr_combinations:
        stochastic_network_analysis(comb)
