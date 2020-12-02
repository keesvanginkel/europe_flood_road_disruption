# -*- coding: utf-8 -*-
"""
Created on 2-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import networkx as nx
import igraph as ig
import pickle
import geopandas as gpd
import pandas as pd
import os
import random
from statistics import mean
import time
import copy
import numpy as np

# save path
save_path = r"D:\COACCH_paper\data\output"

# import files
G_nx = nx.read_gpickle(r"P:\osm_flood\network_analysis\albania\albania_graph.gpickle")
pref_time = gpd.read_file(r"P:\osm_flood\network_analysis\albania\time_pref_routes.shp")
all_aois = pickle.load(open(r"P:\osm_flood\network_analysis\albania\aois_albania.p", "rb"))

# parameters
nr_reps = 100

# convert NetworkX graph to iGraph (identifiers change)
G = ig.Graph.from_networkx(G_nx)
print(ig.summary(G))

# Create a dictionary to convert between the networkx id and igraph id.
# Warning! The ids of igraph are updated when e.g. edges are removed.
vid_lookup = {}
for i in range(len(G.vs)):
    vid_lookup[G.vs()[i]['_nx_name']] = i
print(vid_lookup)


def aoi_combinations(all_aois_list, nr_comb, nr_iterations):
    return [random.choices(all_aois_list, k=nr_comb) for i in range(nr_iterations)]


def stochastic_network_analysis(nr_comb):
    # SKIP IF NR of COMBINATIONS OF AOI IS ALREADY FINISHED BY CHECKING IF OUTPUT FILE IS ALREADY CREATED
    if os.path.exists(os.path.join(save_path, 'aoi_{}.csv'.format(nr_comb))):
        print('{} already finished!'.format(nr_comb))
        return None

    # for only 1 AoI and the maximum nr of AoI's, there is a slightly different approach
    if nr_comb == 1:
        list_aois = all_aois
    elif nr_comb == len(all_aois):
        list_aois = [all_aois]
    else:
        list_aois = aoi_combinations(all_aois, nr_comb, nr_reps)

    # initiate variables
    df = pd.DataFrame(columns=['AoI combinations', 'disrupted', 'avg extra time', 'AoI removed', 'no detour'])
    tot_routes = len(pref_time.index)
    weighing = 'time'

    start = time.time()
    for i, aoi in enumerate(list_aois):
        # Make a copy of the graph
        H = copy.deepcopy(G)

        # remove the edges per flood event (Area of Influence)
        if nr_comb == 1:
            to_remove = G.es.select(lambda e: aoi in e.attributes()['AoI_rp100'])
        else:
            to_remove = G.es.select(lambda e: set(aoi) & set(e.attributes()['AoI_rp100']))
        H.delete_edges(to_remove)

        extra_time = []
        disrupted = 0
        nr_no_detour = 0
        for ii in range(len(pref_time.index)):
            o, d = pref_time.iloc[ii][['o_node', 'd_node']]
            o = vid_lookup[int(o)]
            d = vid_lookup[int(d)]

            # calculate the (alternative) distance between two nodes
            alt_route = H.shortest_paths_dijkstra(source=o, target=d, weights=weighing)
            alt_route = alt_route[0][0]
            if alt_route != np.inf:
                # alt_route = inf if the route is not available
                # append to list of alternative routes to get the average
                extra_time.append(alt_route - pref_time.iloc[ii]['time'])
                if pref_time.iloc[ii]['time'] != alt_route:
                    # the alternative route is different from the preferred route
                    disrupted += 1
            else:
                # append to calculation dataframe
                disrupted += 1
                nr_no_detour += 1

        df = df.append({'AoI combinations': nr_comb, 'disrupted': disrupted / tot_routes * 100,
                        'avg extra time': mean(extra_time), 'AoI removed': aoi, 'no detour': nr_no_detour / tot_routes * 100},
                       ignore_index=True)

        # progress bar
        print(str(round(i / len(list_aois) * 100, 0)) + '%')

    df.to_csv(os.path.join(save_path, 'aoi_{}.csv'.format(nr_comb)))
    end = time.time()
    print('Nr combinations: {}, time elapsed: {}'.format(nr_comb, end - start))


nr_combinations = [1,2,3,4,5,10,20,30,40,50]
for comb in nr_combinations:
    stochastic_network_analysis(comb)
