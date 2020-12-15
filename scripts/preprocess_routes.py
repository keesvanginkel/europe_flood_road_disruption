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
from network import nearest
import pickle
import matplotlib.pyplot as plt
from flow_model import create_graph
import igraph as ig
from statistics import mode
from shapely.geometry import box, mapping
from shapely import wkt
import rioxarray
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
translate_cntr_codes = pd.read_csv(r".\europe_flood_road_disruption\data\country_codes.csv", delimiter=';').set_index('code').to_dict(orient='dict')
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
i=0
current_country = translate_cntr_codes['country'][network_files[i].split('-')[0].split('\\')[-1]].lower()
print("Current iteration is for:", current_country)

# create the folder if it does not exist yet
try:
    os.makedirs(output_folder.format(current_country))
except OSError as e:
    print("Folder already exists, that folder is used:", e)

# read the network files from Elco Koks
network = pd.read_feather(network_files[i])

# create a geometry column with shapely geometries
network['geoms'] = pyg.io.to_wkt(pyg.from_wkb(network.geometry))
network['geoms'] = network['geoms'].apply(wkt.loads)
network.drop('geometry', axis=1, inplace=True)
network.rename(columns={'geoms': 'geometry'}, inplace=True)

# Create a GeoDataFrame from the pandas DataFrame with the CRS EPSG:4326
network_geoms = gpd.GeoDataFrame(network, geometry='geometry', crs='EPSG:4326')
network_geoms = network_geoms.to_crs('EPSG:3857')
network_geoms = network_geoms.loc[~network_geoms['highway'].isin(['tertiary', 'tertiary_link'])]

pref_routes = gpd.read_file(r"D:\COACCH_countries\albania\time_pref_routes.shp")
all_aois = pickle.load(open(r"D:\COACCH_countries\aois_lists\aois_albania.p", "rb"))

pref_routes.columns
all_aois

# needed in the optimal routes dataframe: o_node, d_node (origin and destination node ID's)
# time it takes to drive the optimal route from that origin to destination
for ii in range(len(pref_time.index)):
    o, d = pref_time.iloc[ii][['o_node', 'd_node']]
    pref_time.iloc[ii]['time']

# save OD matrix as pandas feather datatype
# save the same matrix still also as shp for quick visualisation




possibleOD['geometry'] = possibleOD['geometry'].apply(pyg.from_wkt)

def prepare_possible_OD(gridDF, nodes, tolerance=1):
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
    nodeIDs = []
    sindex = pyg.STRtree(nodes['geometry'])

    pos_OD_nodes = []
    # pos_tot_pop = []
    for i in gridDF.itertuples():
        ID = nearest(i.geometry, nodes, sindex, tolerance)
        # If a node was found
        if ID > -1:
            pos_OD_nodes.append(ID)
            # pos_tot_pop.append(i.tot_pop)
    a = nodes.loc[nodes.id.isin(pos_OD_nodes)]
    # Create a geopackage of the possible ODs
    # with Geopackage('nodyBGR.gpkg', 'w') as out:
    #    out.add_layer(a, name='finanod', crs='EPSG:4326')
    nodes = np.array([pos_OD_nodes])
    node_unique = np.unique(nodes)
    # count = np.array([pos_tot_pop])

    # List comprehension to add total populations of recurring nodes
    final_possible_pop = [i for i in node_unique]
    return final_possible_pop


# get only the nuts regions in the selected countries
if len(incl_nuts) > 0:
    nuts_selection = nuts.loc[(nuts['CNTR_CODE'].isin(country_codes)) & (~nuts['NUTS_ID'].isin(excl_nuts)) &
                              (nuts['NUTS_ID'].isin(incl_nuts))]
else:
    nuts_selection = nuts.loc[(nuts['CNTR_CODE'].isin(country_codes)) & (~nuts['NUTS_ID'].isin(excl_nuts))]

# get centroids from NUTS-3 regions
nuts_selection['centroid'] = nuts_selection['geometry'].centroid

centroids = nuts_selection[['LEVL_CODE', 'NUTS_ID', 'CNTR_CODE', 'NUTS_NAME', 'FID', 'centroid']]
centroids = gpd.GeoDataFrame(centroids, geometry='centroid')

# reproject to 4326
centroids.crs = {'init': 'epsg:3035'}
centroids = centroids.to_crs({'init': 'epsg:4326'})

# find closest vertice of road network to centroid
# create dictionary of the roads geometries
edge_list = [e for e in graph.edges.data(keys=True) if 'geometry' in e[-1]]
vertices_dict = {}
for line in edge_list:
    vertices_dict[(line[0], line[1], line[2])] = [Point(p) for p in set(list(line[-1]['geometry'].coords))]

# create list of all points to search in
all_vertices = [p for sublist in list(vertices_dict.values()) for p in sublist]

# create an empty spatial index object to search in
idx = rtree.index.Index()

# populate the spatial index
for i, pnt in enumerate(all_vertices):
    idx.insert(i, pnt.bounds)

ODs = []
match_ids = []
for i in range(len(centroids.index)):
    c = centroids['centroid'].iloc[i]

    # find the closest vertice and line the vertice lays on
    target = list(idx.nearest(c.bounds))

    # draw a progress bar
    drawProgressBar(i / len(centroids))

    # do nothing if there is no target point
    if not target:
        warnings.warn("No vertex to place the Origin/Destination node on for centroid with NUTS_ID ".format(
            centroids['NUTS_ID'].iloc[i]))
        continue

    points_list = [all_vertices[ip] for ip in target]

    distance_list = [(p, c.distance(p)) for p in points_list if c.distance(p) > 0]
    closest_point, closest_distance = min(distance_list, key=lambda t: t[1])

    # check on which road this point lays
    road_i = getKeysByValue(vertices_dict, closest_point)
    match_ids.append(graph[road_i[0]][road_i[1]][road_i[2]]['match_id'])
    # save in list
    ODs.append(points_list[0])

# save in dataframe
centroids['OD'] = ODs
centroids['match_ids'] = match_ids

# save the road vertices closest to the centroids as geometry, delete the centroid geometry
centroids = gpd.GeoDataFrame(centroids, geometry='OD')
centroids = centroids.drop(columns=['centroid'])
centroids.crs = {'init': 'epsg:4326'}

# save OD points
if save_shp:
    gdf_to_shp(centroids, os.path.join(file_output, name + "_OD.shp"))
    print("Saved centroids of {} to shapefile: {}".format(os.path.join(file_output, name + "_OD.shp"), name))
if save_pickle:
    pickle.dump(centroids, open(os.path.join(file_output, name + "_OD.p"), 'wb'))
    print("Saved centroids of {} to pickle: {}".format(os.path.join(file_output, name + "_OD.p"), name))


