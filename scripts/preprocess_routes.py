# -*- coding: utf-8 -*-
"""
Created on 15-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import os, sys

sys.path.append(r"D:\COACCH_paper\trails-master\src\trails")

# folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(folder)
# os.chdir(os.path.dirname(folder))  # set working directory to top folder

import pygeos as pyg
import pandas as pd
import geopandas as gpd
from network import nearest, graph_load
import itertools
import feather
from shapely import wkt
import numpy as np

# define in- and output folders
input_folder = r"D:\COACCH_paper\data"
output_folder = r"D:\COACCH_paper\data\output\{}"

# NUTS-3 regions of Europe
nuts_3_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_3.shp"

# Location of graphs of all countries in Europe, saved in *.feather format
networks_europe_path = os.path.join(input_folder, 'networks_europe_elco_koks')
network_files = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if f.endswith('-edges.feather')]

# Get the country codes from the *.feather files (networks) and see with a country code
# translation table which countries are there
country_codes = [f.split('-nodes')[0] for f in os.listdir(networks_europe_path) if f.endswith('-nodes.feather')]
translate_cntr_codes = pd.read_csv(r".\europe_flood_road_disruption\data\country_codes.csv", delimiter=';').set_index('code3').to_dict(orient='dict')
print(len(country_codes), 'countries:\n' + '\n'.join([translate_cntr_codes['country'][cntry] for cntry in country_codes]))

# Create a CSV file containing the coordinates of the centroids of the NUTS-3 regions, its country code ('CNTR_CODE'),
# nuts code ('NUTS_ID') and nuts name ('NUTS_NAME')
def create_centroids_csv(path_to_shp, discard_fields, save_feather):
    nuts = gpd.read_file(path_to_shp)
    for fld in discard_fields:
        del nuts[fld]

    # get centroids from NUTS-3 regions
    nuts['centroids_geom'] = nuts['geometry'].centroid
    centroids = gpd.GeoDataFrame(nuts, geometry='centroids_geom')
    del centroids['geometry']

    # reproject to 4326
    centroids.crs = 'epsg:3035'
    centroids = centroids.to_crs('epsg:4326')

    wkt_array = gpd.array.to_wkt(centroids.centroids_geom.values)
    df = pd.DataFrame(centroids.drop(columns='centroids_geom'))
    df['geometry'] = wkt_array

    df.to_feather(save_feather)
    print("Saved in", save_feather)


create_centroids_csv(nuts_3_regions, ['LEVL_CODE', 'FID'], os.path.join(input_folder, 'output', 'europe_nuts3_centroids.feather'))

centroids = pd.read_feather(os.path.join(input_folder, 'output', 'europe_nuts3_centroids.feather'))

# iterate over the graphs
i = 0
cntry_code = network_files[i].split('-')[0].split('\\')[-1]
current_country = translate_cntr_codes['country'][cntry_code].lower()
print("Current iteration is for:", current_country)

# create the folder if it does not exist yet
try:
    os.makedirs(output_folder.format(current_country))
except OSError as e:
    print("Folder already exists, that folder is used:", e)

# select the centroids that are in the country that is analysed
selected_centroids = centroids.loc[centroids['CNTR_CODE'] == translate_cntr_codes['code2'][cntry_code]]
selected_centroids['geometry'] = selected_centroids['geometry'].apply(pyg.from_wkt)

# read the network files from Elco Koks
network = pd.read_feather(network_files[i])
G = graph_load(network)

# # create a geometry column with shapely geometries
# network['geoms'] = pyg.io.to_wkt(pyg.from_wkb(network.geometry))
# network['geoms'] = network['geoms'].apply(wkt.loads)
# network.drop('geometry', axis=1, inplace=True)
# network.rename(columns={'geoms': 'geometry'}, inplace=True)
#
# # Create a GeoDataFrame from the pandas DataFrame with the CRS EPSG:4326
# network_geoms = gpd.GeoDataFrame(network, geometry='geometry', crs='EPSG:4326')
# network_geoms = network_geoms.to_crs('EPSG:3857')
# network_geoms = network_geoms.loc[~network_geoms['highway'].isin(['tertiary', 'tertiary_link'])]



# read nodes
nodes = feather.read_dataframe(network_files[i].replace("-edges", "-nodes"))
nodes.geometry = pyg.from_wkb(nodes.geometry)

node_ids_od_pairs = prepare_possible_OD_EU(selected_centroids, nodes, tolerance=0.1)
for n_id, nuts3_code in node_ids_od_pairs:
    nodes.loc[nodes['id'] == n_id, 'nuts3'] = nuts3_code

# dataframe to save the preferred routes
pref_routes = gpd.GeoDataFrame(columns=['o_node', 'd_node', 'origin', 'destination',
                                        'pref_path', 'time', 'match_ids', 'geometry'],
                               geometry='geometry', crs={'epsg:4326'})  #'AoIs',

# create the routes between all OD pairs
for o, d in itertools.combinations(node_ids_od_pairs, 2):
    alt_route = H.shortest_paths_dijkstra(source=o, target=d, weights=weighing)
    alt_route = alt_route[0][0]

    # calculate the length of the preferred route
    pref_route = nx.dijkstra_path_length(graph, o[0], d[0], weight=weighing_name)

    # save preferred route nodes
    pref_nodes = nx.dijkstra_path(graph, o[0], d[0], weight=weighing_name)

    # found out which edges belong to the preferred path
    edgesinpath = zip(pref_nodes[0:], pref_nodes[1:])

    pref_edges = []
    match_list = []
    aoi_list = []
    for u, v in edgesinpath:
        edge = sorted(graph[u][v], key=lambda x: graph[u][v][x][weighing_name])[0]
        if 'geometry' in graph[u][v][edge]:
            pref_edges.append(graph[u][v][edge]['geometry'])
        if 'match_id' in graph[u][v][edge]:
            match_list.append(graph[u][v][edge]['match_id'])
        if 'AoI_rp100' in graph[u][v][edge]:
            if isinstance(graph[u][v][edge]['AoI_rp100'], list):
                aoi_list.extend(graph[u][v][edge]['AoI_rp100'])
            else:
                aoi_list.append(graph[u][v][edge]['AoI_rp100'])

    # remove al 0's from the AoI list
    aoi_list = [float(x) for x in aoi_list if (x != 0) and (x == x)]
    aoi_list = list(set(aoi_list))
    pref_edges = MultiLineString(pref_edges)
    pref_routes = pref_routes.append({'o_node': o[0], 'd_node': d[0], 'origin': o[1],
                                      'destination': d[1], 'AoIs': aoi_list, 'pref_path': pref_nodes,
                                      weighing_name: pref_route, 'match_ids': match_list,
                                      'geometry': pref_edges}, ignore_index=True)
if save_shp:
    gdf_to_shp(pref_routes, os.path.join(file_output, '{}_pref_routes.shp'.format(name)))
    print("Preferred routes saved to {}".format(os.path.join(file_output, '{}_pref_routes.shp'.format(name))))

if save_pickle:
    pref_routes[['origin', 'destination', 'AoIs', 'pref_path', weighing_name,
                 'match_ids']].to_pickle(os.path.join(file_output, '{}_pref_routes.pkl'.format(name)))
    print("Preferred routes saved to {}".format(os.path.join(file_output, '{}_pref_routes.pkl'.format(name))))



pref_routes = gpd.read_file(r"D:\COACCH_countries\albania\time_pref_routes.shp")
pref_routes.columns

# needed in the optimal routes dataframe: o_node, d_node (origin and destination node ID's)
# time it takes to drive the optimal route from that origin to destination
for ii in range(len(pref_time.index)):
    o, d = pref_time.iloc[ii][['o_node', 'd_node']]
    pref_time.iloc[ii]['time']

# save OD matrix as pandas feather datatype
# save the same matrix still also as shp for quick visualisation


def prepare_possible_OD_EU(gridDF, nodes, tolerance=1):
    """Returns an array of tuples, with the first value the node ID to consider, and the
       second value the total population associated with this node.
       The tolerance is the size of the bounding box to search for nodes within
    Args:
        gridDF (pandas.DataFrame): A dataframe with the grid centroids and their population
        nodes (pandas.DataFrame): A dataframe of the road network nodes
        tolerance (float, optional): size of the bounding box . Defaults to 0.1.
    Returns:
        final_possible_pop (list): a list of tuples representing the nodes and their population

    Adjusted from https://github.com/BenDickens/trails/blob/master/src/trails/network.py#L179 to work
    without population data.
    """
    sindex = pyg.STRtree(nodes['geometry'])

    pos_OD_nodes = []
    pos_nuts3 = []
    for i in gridDF.itertuples():
        ID = nearest(i.geometry, nodes, sindex, tolerance)
        # If a node was found
        if ID > -1:
            pos_OD_nodes.append(ID)
            pos_nuts3.append(i.NUTS_ID)
    # a = nodes.loc[nodes.id.isin(pos_OD_nodes)]
    # Create a geopackage of the possible ODs
    # with Geopackage('nodyBGR.gpkg', 'w') as out:
    #    out.add_layer(a, name='finanod', crs='EPSG:4326')
    nodes = np.array([pos_OD_nodes])
    node_unique = np.unique(nodes)
    nuts3_codes = np.array([pos_nuts3])

    # List comprehension to add total populations of recurring nodes
    final_possible_pop = [(ii, nuts3_codes[nodes == ii][0]) for ii in node_unique]
    return final_possible_pop


