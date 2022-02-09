# -*- coding: utf-8 -*-
"""
Created on 15-12-2020

@author: Frederique de Groen & Kees van Ginkel

Part of a COACCH percolation analysis of European road networks.

"""

#Todo: rename paths in this folder


import os, sys

#sys.path.append(r"P:\osm_flood\network_analysis\igraph\trails\src\trails")
sys.path.append(r"D:\Python\trails\src\trails")

# folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(folder)
# os.chdir(os.path.dirname(folder))  # set working directory to top folder

import igraph as ig
import pygeos as pyg
import pandas as pd
import geopandas as gpd
from network import nearest #trails import
#from snkit.network import nearest

import itertools
import feather
import json
from pathlib import Path
from shapely import wkt
from shapely.geometry import MultiLineString
import numpy as np
from numpy import object as np_object
from tqdm import tqdm
from pathos.multiprocessing import Pool
import warnings
import logging

from utils import load_config
from Europe_utils import NUTS_2_islands, NUTS_3_remote,NUTS_3_islands

#Suppress pandas SettingWithCopyWArning
pd.options.mode.chained_assignment = None  # default='warn'

# NUTS-3 regions of Europe
# nuts_3_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_3.shp"
# nuts_2_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_2.shp"

# list of islands that aren't assessed
#Todo: combine these with the list of ignored regions in the Europe_utils!
#Update this list with Elco's new graphs

# overseas = ['PT200', 'PT300',  # Portugal: Azores and Madeira
#             'ES531', 'ES532', 'ES533', 'ES703', 'ES704', 'ES705', 'ES706', 'ES707', 'ES708', 'ES709', 'ES630', 'ES640',  # Spain: Canary Islands
#             'FRY10', 'FRY20', 'FRY30', 'FRY40',  # France: overseas areas: Gouadeloupe, Martinique, French Guiana, La Reunion, Mayotte "])
#             'FRY50', 'FRM01', 'FRM02',
#             'EL307', 'EL411', 'EL412', 'EL413', 'EL421', 'EL422', 'EL431', 'EL432', 'EL433', 'EL434', 'EL621', 'EL622', 'EL623',  # Greece
#             'HR037',  # croatia
#             'UKN0',  # UK (NUTS-2 classification)
#             'DK014',  # Denmark
#             'DK014','DK022','DK021','DK011','DK012','DK013',  # Denmark #tempfix 26/4. Probably we shouldn't exclude Cophenhagen though...
#             'FI200',  # Finland
#             'SE214'  # Sweden
#              ]

# Create a CSV file containing the coordinates of the centroids of the NUTS-3 regions, its country code ('CNTR_CODE'),
# nuts code ('NUTS_ID') and nuts name ('NUTS_NAME')
def create_centroids_csv(path_to_shp, discard_fields, save_feather):
    """
    Creates centroids file from shapefile with NUTS-regions

    :param path_to_shp (string) : Path to shapefile
    :param discard_fields: fields that will be ignored
    :param save_feather (string) : Path to feather file to save the results
    :return: dataframe with centroids per region
    """

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

    return df

#TO CREATE CENTROIDS_CSV
# create_centroids_csv(nuts_2_regions, ['LEVL_CODE', 'FID'], os.path.join(input_folder, 'europe_nuts2_centroids.feather'))

# CREATE CENTROIDS FOR BENELUX (UNCERTAINTY ANALYSIS)
#benelux_area = Path(r'D:\Europe_percolation\uncertainty_analysis\Benelux\benelux.shp')
#benelux_centroids = load_config()['paths']['data'] / 'benelux_nuts3_centroids.feather'
#benelux = create_centroids_csv(benelux_area, ['LEVL_CODE', 'FID'], benelux_centroids)

# CREATE CENTROIDS FOR NORWAY (ONE SHIFTED CENTROID)
# manually shifted one position in QGIS
# benelux_centroids_shp = load_config()['paths']['data'] / 'nuts3_centroids_norway.shp'
# gdf = gpd.GeoDataFrame.from_file(benelux_centroids_shp)
# df = pd.DataFrame(gdf)
# wkt_array = gpd.array.to_wkt(df.geometry.values)
# df['geometry'] = wkt_array
# df.to_feather((load_config()['paths']['data'] / 'nuts3_centroids_norway.feather'))

#benelux = create_centroids_csv(benelux_area, ['LEVL_CODE', 'FID'], benelux_centroids)

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

    Issue (Kees, 26/4/2021): sometimes returns a node that is not connected to the graph (i.e. has degree 0)
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


# Creates a graph
def graph_load(edges, save_columns):
    """Creates

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns
        save_columns (list of str): a list of the names of the columns that should be added to the graph

    Returns:
        igraph.Graph (object) : a graph with distance, time, and other 'colname' attributes

    Adjusted from https://github.com/BenDickens/trails/blob/42c11018885baeee047003d436b13bd6473bfb0e/src/trails/network.py#L65
    to also save other attributes in the graph.
    """
    # return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    graph = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id), max(edges.to_id))
    graph.add_vertices(max_node_id + 1)
    edge_tuples = zip(edges.from_id, edges.to_id)
    graph.add_edges(edge_tuples)
    graph.es['distance'] = edges.distance
    graph.es['time'] = edges.time
    for colname in save_columns:
        graph.es[colname] = edges[colname]
    return graph


def gdf_to_shp(gdf, result_shp):
    """Takes in a geodataframe object and outputs shapefiles at the paths indicated by edge_shp and node_shp

    Arguments:
        gdf [geodataframe]: geodataframe object to be converted
        edge_shp [str]: output path including extension for edges shapefile
        node_shp [str]: output path including extension for nodes shapefile
    Returns:
        None
    """
    for col in gdf.columns:
        if gdf[col].dtype == np_object and col != gdf.geometry.name:
            gdf[col] = gdf[col].astype(str)

    gdf.to_file(result_shp, driver='ESRI Shapefile', encoding='utf-8')



def log_preprop_optimal_routes(cntry,nuts_class,weighing,config):
    """
    Create a logger for optimal_routes()

    """
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(funcName)20s()] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger("{}-{}-{}".format(__name__,cntry,nuts_class,weighing))
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    #Add file handler
    logger_filename = config['paths']['logs'] / 'preprocessing' / 'log_{}_{}_{}_{}.log'.format(
        'PreProc_OptimalRoutes', cntry, nuts_class, weighing)
    fileHandler = logging.FileHandler(logger_filename)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    logger.info('Logger is created for optimal_routes()')
    return logger

def from_edge_geoms_to_node_geoms(edge_network,logger = None):
    """
    Infer the coordinates of the nodes from a DataFrame that describes the coordinates of the edges as Linestrings.
    This assumes that coords in the linestring are in the direction 'from' to 'to' the edge ID
    A simple check will be carried out to verify this assumption

    Arguments:
        *edge_network* (Pandas DataFrame) :
            This df should have the columns:
                *from_id* (float/int) : containing the indices of the 'from' nodes
                *to_id* (float/int) : contains the indices of the 'to' nodes
                *geometry* (Shapely linestrings) : describing the geometry of the edge

    Returns:
        *result* (dict) : keys are the node IDs; values are shapely Points with the coordinates

    """
    from shapely.geometry import Point
    #identify the begin and end point of each linestring object
    #edge_network['boundary'] = edge_network['geometry'].apply(lambda x: x.boundary if len(x.coords) > 2 else None)
            #the .boundary option does not work for linestrings with len 2
    #edge_network['boundary'] = edge_network['geometry'].apply(lambda x: (x.coords[0],x.coords[-1]))
    edge_network['from_point_geom'] = edge_network['geometry'].apply(lambda x: Point(x.coords[0]))
    edge_network['to_point_geom'] = edge_network['geometry'].apply(lambda x: Point(x.coords[-1]))

    result = {}  # keys : point_ids, values: shapely geometry points

    #so one list of (id,geom)-pairs infered from the 'from points'
    point_geoms_froms = edge_network[['from_id', 'from_point_geom']].values
    # and one infered from the 'to_points'
    point_geoms_tos = edge_network[['to_id', 'to_point_geom']].values

    #Check for possible conflicts between the multiple linestrings from which the same id could be inferred
    for (id, geom) in point_geoms_froms: #start by iterating over the from points
        if not id in result.keys():
            result[id] = geom
        else:
            if not result[id] == geom: #if not exactly the same as existing
                report_conflict(id,result[id],geom,logger=logger) #report the difference

    for (id, geom) in point_geoms_tos: #start by iterating over the from points
        if not id in result.keys(): #if not already exists: save the geometry as end result
            result[id] = geom
        else: #if already exists:
            if not result[id] == geom: #if not exactly the same as existing
                report_conflict(id,result[id],geom,logger=logger) #report the difference

    result = dict(sorted(result.items())) #sort the keys before returning
    return result

def report_conflict(id,already_saved_geom,conflicting_geom,logger=None):
    """Compare conflicting geometries
     # The geometry that was already inferred from another direction
     # The new geom that apparently is not exactly the same

     Raises warning / write to log whether the difference is major or minor

     """
    # try to calculate distance:
    distance_between_geoms = already_saved_geom.distance(conflicting_geom)
    if distance_between_geoms < 100:  # displacement is less than 100 m
        warning = 'Minor conflicting geoms for id = {}, distance between '.format(id) + \
                  'original and conflicting geom = {} m'.format(distance_between_geoms)
    else:
        warning = 'Major conflicting geoms for id = {}, distance between '.format(id) + \
                  'original and conflicting geom = {} m'.format(distance_between_geoms)
    if not logger == None:
        logger.warning(warning)
    else:
        warnings.warn(warning)
    return warning




def optimal_routes(cntry,nuts_class = 'nuts3',weighing = 'time',config_file='config.json',special_setting=None):
    """
    Preprocessing: finds the optimal routes between the NUTS-3 or NUTS-2 regions in a country

    Parameters:
     *cntry* (string) : 3-letter code of the country
     *nuts_class* (string) : 'nuts2' or 'nuts3' (default)
     *weighing* (string) : 'time' (or distance?)
     *special_settig* (string) : 'Flanders', 'Wallonia', 'Benelux', 'Rhine-alpine', 'shifted_centroids', None (default)
        Used for running the script with other than default settings

    :return:

    Requires
     - feather files with edges and nodes of the road network of the country: in output_folder
     - feather file with the centroids of the NUTS2 or NUTS-3 regions in the country
    """
    config = load_config(file=config_file)

    # INSTANTIATE LOGGING FUNCTIONALITY
    logger = log_preprop_optimal_routes(cntry, nuts_class, weighing,config=config)
    logger.info('Logging optimal_routes() for cntry "{}", nuts_class "{}", weighing "{}"'.format(cntry,nuts_class,weighing))

    try:
        #Some very specific settings, only used when doing uncommon things, such as the
                                     # uncertainty analysis
        if special_setting != None:
            logger.warning("Running in non-default mode, with special setttings '{}'".format(special_setting))

        if special_setting in ['Flanders','Wallonia']:
            from Europe_utils import NUTS_3_Flanders, NUTS_3_Wallonia

        if weighing != 'time':
            logger.warning("Running in non-default mode. The weighing of routes is not done by time, but by '{}'.".format(weighing))


        logger.info('Using config file: "{}"'.format(config_file))

        # define in- and output folders
        output_folder = config['paths']['preproc_output']

        # Location of graphs of all countries in Europe, saved in *.feather format
        networks_europe_path = config['paths']['graphs_folder']
        edge_file = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if
                     f == cntry + '-edges.feather'][0]

        # Get the country codes from the *.feather files (networks) and see with a country code
        # translation table which countries are there
        country_codes = config['paths']['data'] / 'country_codes.csv'
        assert country_codes.exists
        translate_cntr_codes = pd.read_csv(country_codes, delimiter=';').set_index('code3').to_dict(orient='dict')

        current_country = translate_cntr_codes['country'][cntry].lower()
        if current_country.endswith(' '): #Remove space in last character of country name (if any)
            current_country = current_country[:-1]
        print("\nCurrent iteration is for:", current_country)

        country_dir = output_folder / current_country
        od_matrix = country_dir / 'optimal_routes_{}_{}_{}.feather'.format(weighing, current_country,nuts_class)

        # create the folder if it does not exist yet
        if not country_dir.exists() : country_dir.mkdir()

        if od_matrix.exists():
            logger.warning('Optimal_routes(): Country = {} already finished! Stops function.'.format(current_country))
            return None

        # The centroids of the NUTS-3/NUTS-2 regions
        centroids = config['paths']['data'] / 'europe_{}_centroids.feather'.format(nuts_class)
        if special_setting == 'Benelux':
            centroids = config['paths']['data'] / 'benelux_nuts3_centroids.feather'
        elif special_setting == 'Rhine-alpine':
            centroids = config['paths']['data'] / 'rhine_alphine_ods.feather'
        elif special_setting == 'shifted_centroids':
            centroids = config['paths']['data'] / 'belgium_shifted_centroids.feather'
        if special_setting == 'Norway_relocate':
            centroids = config['paths']['data'] / 'nuts3_centroids_norway.feather'

        centroids = pd.read_feather(centroids)

        # select the centroids that are in the country that is analysed
        # discard the centroids of the small islands that are not reachable from mainland Europe
        if nuts_class == 'nuts3':
            overseas = NUTS_3_remote(overseas=True,Creta=True,Spain=True)
            overseas.extend(NUTS_3_islands())
            logger.info('Removing the following NUTS-regions {}'.format(
                [c for c in overseas if c.startswith(translate_cntr_codes['code2'][cntry])]))
            selected_centroids = centroids.loc[(centroids['CNTR_CODE'] == translate_cntr_codes['code2'][cntry]) &
                                           (~centroids['NUTS_ID'].isin(overseas))]

            if special_setting == 'Flanders':
                flanders = NUTS_3_Flanders()
                selected_centroids = selected_centroids.loc[selected_centroids['NUTS_ID'].isin(flanders)]

            elif special_setting == 'Wallonia':
                wallonia = NUTS_3_Wallonia()
                selected_centroids = selected_centroids.loc[selected_centroids['NUTS_ID'].isin(wallonia)]




        if nuts_class == 'nuts2':
            overseas = NUTS_2_islands()
            selected_centroids = centroids.loc[(centroids['CNTR_CODE'] == translate_cntr_codes['code2'][cntry]) &
                                           (~centroids['NUTS_ID'].isin(overseas))]
            #print(list(selected_centroids['NUTS_ID'].values))

        if special_setting == 'Rhine-alpine':
            selected_centroids = centroids.copy()

        if special_setting == 'shifted_centroids':
            selected_centroids = centroids.copy()

        logger.info('{} selected centroids for OD-routing: \n {}'.format(
            len(selected_centroids),selected_centroids["NUTS_ID"].values))

        selected_centroids['geometry'] = selected_centroids['geometry'].apply(pyg.from_wkt)

        # read the network files (TRAILS output, with the flood information added)
        network = pd.read_feather(edge_file)
        # network['geometry'] = network['geometry'].apply(wkt.loads)

        # create a geometry column with shapely geometries
        network['geoms'] = pyg.io.to_wkt(pyg.from_wkb(network.geometry))  # see if this should be activated with the new feather files
        network['geoms'] = network['geoms'].apply(wkt.loads)
        network.drop('geometry', axis=1, inplace=True)
        network.rename(columns={'geoms': 'geometry'}, inplace=True)

        # create the graph
        G = graph_load(network, ['geometry', 'id', 'RP100_cells_intersect', 'RP100_max_flood_depth',
                                 'AoI_RP100y_majority', 'AoI_RP100y_unique', 'fds_majority', 'fds__unique'])

        # read nodes
        #Todo: this is possible not necessary?
        #nodes = feather.read_dataframe(edge_file.replace("-edges", "-nodes"))
        #nodes.geometry = pyg.from_wkb(nodes.geometry)

        #Check if the graph has vertices with degree 0
        unconnected = len(G.vs.select(_degree=0))
        if unconnected > 0:
            logger.warning('Graph has {} vertices with degree 0, they are unconnected to the rest of the network. \
            Issue is probably caused by trails'.format(unconnected))

        # find the nodes that are closest to the centroids of the NUTS-3 regions
        #Todo: replace the below function with something that only depends on the edge file.
        inferred_nodes = from_edge_geoms_to_node_geoms(network,logger)
        inferred_nodes_ids = list(inferred_nodes.keys())
        if not len(inferred_nodes_ids) == len(G.vs):
            logger.warning("""The length of the vertex sequence (G.vs) = {} 
                            does not match the length of the inferred nodes = {}
                            This means that many nodes are actually not connected to the graph, probably due to an
                            issue in TRAILS! We will try to work around that issue here.""".format(
                                        len(G.vs),len(inferred_nodes_ids)))
            connected = len(G.vs)-unconnected
            logger.warning('Connected vertices: len(G.vs) - #unconnected vertices = {}'.format(connected))
            if connected == len(inferred_nodes_ids):
                logger.info('Good news: number of connected nodes matches the number of ids for which we found coordinates')
            else:
                logger.warning('Also when disregarding the unconnected nodes, still no match between connected nodes and ids with coordinates')

            dummy_ids = list(range(0,len(G.vs)))
            dummy_geoms = [None] * len(G.vs)
            for (id, geom) in inferred_nodes.items():
                dummy_geoms[id] = geom
            inferred_nodes_ids = dummy_ids
            inferred_nodes_geometries = dummy_geoms
        else:
            assert inferred_nodes_ids == sorted(inferred_nodes_ids)
            inferred_nodes_geometries = list(inferred_nodes.values())

        #Assign the inferred nodes to the vertices of the graph
        G.vs['id'] = inferred_nodes_ids
        G.vs['geometry'] = inferred_nodes_geometries

        #dummy file to imitate the old prepare_possible_OD_EU() input
        new_nodes = pd.DataFrame(data=inferred_nodes_geometries,index=inferred_nodes_ids,columns=['geometry'])
        new_nodes['id'] = list(new_nodes.index)
        new_nodes = pd.DataFrame(gpd.GeoDataFrame(new_nodes,geometry='geometry',crs='epsg:3035').to_crs('epsg:4326')) #to wgs84
        new_nodes['geometry'] = new_nodes['geometry'].apply(lambda x: pyg.io.from_shapely(x))

        nodes = new_nodes
        node_ids_od_pairs = prepare_possible_OD_EU(selected_centroids, nodes, tolerance=0.5)  # was 0.1
        if len(node_ids_od_pairs) != len(selected_centroids):
            logger.warning("Script could not find corresponding node/vertex for all NUTS-regions")
            missing_nuts = [n for n in list(selected_centroids.NUTS_ID.unique()) if n not in list(zip(*node_ids_od_pairs))[1]]
            logger.warning("NUTS-regions excluded in analysis: {}".format(print(missing_nuts)))

        #Save the results as a json-file
        ids_to_nuts = dict([(nuts, id.item()) for (id, nuts) in node_ids_od_pairs]) #keys: nuts-code; value = vertex id in the graph
        outfile = country_dir / (cntry + '_' + nuts_class + '__vertexID_to_nuts.json')
        with open(outfile, 'w') as f:
            json.dump(ids_to_nuts, f)
        logger.info('Save lookup table from vertex/nodeID to nuts as json file in: {}'.format(outfile))

        #Temporary fix: add the nuts-number to the matching node ids
        for n_id, nuts_code in node_ids_od_pairs:
            nodes.loc[nodes['id'] == n_id, nuts_class] = nuts_code  # Can be nuts2 or nuts3
        G.vs[nuts_class] = nodes[nuts_class] #Assign nutslabels to graph

        #Todo: we can also simplify the graph, but not 100% wether that could mess up the indexing...
        #G.vs.select(_degree=0).delete()
        #logger.warning('Tries to delete all unconnected edges, will save this new graph')


        #Check if the number of vertices with nuts-data matched the number of centroids
        assert len([v for v in G.vs if not pd.isna(v[nuts_class])]) == len(selected_centroids)


        #Not sure if this part is still needed
        #nodes.geometry = pyg.to_wkb(nodes.geometry)
        #nodes.to_feather(edge_file.replace("-edges", "-nodes_{}".format(nuts_class)))

        # save the list of combinations that should be ran in the excel sheet that is also used to check whether
        # the number of optimal routes is correct
        all_aois = list(set([int(x) for l in list(network['AoI_RP100y_unique']) for x in l if (x != 0) and (x == x)]))
        max_aoi = len(all_aois)

        list_combinations = new_list_combinations(max_aoi) #todo: sometimes this returns a zero, this should not be possible!

        #Todo: make sure it does not add empty columns
        combinations_csv = config['paths']['data'] / "{}_combinations.csv".format(nuts_class)
        df = pd.read_csv(combinations_csv,sep=';',index_col=0)
        nr_optimal_routes = df.loc[df['code3'] == cntry, 'nr_routes'].iloc[0]
        df.loc[df['code3'] == cntry, 'aoi_combinations'] = " ".join(list_combinations)
        df.to_csv(combinations_csv,sep=';')

        # dataframe to save the optimal routes
        pref_routes = gpd.GeoDataFrame(columns=['o_node', 'd_node', 'origin', 'destination', 'AoIs',
                                                'v_ids', weighing, 'e_ids', 'geometry'],
                                       geometry='geometry', crs='epsg:3035')


        # create the routes between all OD pairs
        for o, d in tqdm(itertools.combinations(node_ids_od_pairs, 2), desc='NUTS optimal routes {}'.format(current_country)):
            # calculate the length of the optimal route
            # now the graph is treated as directed, make mode=ig.ALL to treat as undirected

            if G.degree(o[0]) == 0:
                raise ValueError('Origin node/vertex {} has degree 0, meaning that it is not connected to the graph'.format(o))
            if G.degree(d[0]) == 0:
                raise ValueError('Destination node/vertex {} has degree 0, meaning that it is not connected to the graph'.format(d))


            #This way of calculating the routes is much less efficient
            optimal_route = G.shortest_paths_dijkstra(source=o[0], target=d[0], mode=ig.OUT, weights=weighing)

            # Get the nodes and edges of the shortest path.
            # Its not very clear from documentation but if I'm correct this function also uses Dijkstra's algorithm
            # Comparing the outcomes of the shortest_paths_dijkstra and the weighing of the edges that are taken in the route,
            # are different in the far decimal numbers, but when rounding they are the same.

            path_nodes = G.get_shortest_paths(o[0], to=d[0], weights=weighing, mode=ig.OUT, output="vpath")
            path_edges = G.get_shortest_paths(o[0], to=d[0], weights=weighing, mode=ig.OUT, output="epath")

            aoi_list = 0
            # Create a list of AoI's the route crosses
            if 'AoI_RP100y_unique (e)' in G.summary():
                # remove al 0's and nan's from the AoI list
                aoi_list = list(set([int(x) for l in G.es[path_edges[0]]['AoI_RP100y_unique'] for x in l if (x != 0) and (x == x)]))

            if 'geometry (e)' in G.summary():
                edges_geoms = MultiLineString(G.es[path_edges[0]]['geometry'])

            if 'id (v)' in G.summary():
                path_nodes_ids = G.vs[path_nodes[0]]['id']

            if 'id (e)' in G.summary():
                path_edges_ids = G.es[path_edges[0]]['id']

            pref_routes = pref_routes.append({'o_node': o[0], 'd_node': d[0], 'origin': o[1],
                                              'destination': d[1], 'AoIs': aoi_list, 'v_ids': path_nodes_ids,
                                              weighing: optimal_route[0][0], 'e_ids': path_edges_ids,
                                              'geometry': edges_geoms}, ignore_index=True)

        if len(pref_routes.index) != nr_optimal_routes:
            logger.warning("Warning. The number of preferred routes does not match with the \
                           number set in the {}. Check for country {}".format(combinations_csv,cntry))

        # remove impossible routes (with time = inf)
        pref_routes = pref_routes.replace([np.inf, -np.inf], np.nan).dropna(subset=[weighing])

        # Save optimal routes as shapefile, pickle and feather (feather is used in the percolation_optimized.py)
        # Probably, sometimes it cannot be saved as a shapefile is the route is too long or whatsoever.
        gdf_to_shp(pref_routes, (country_dir / 'optimal_routes_{}_{}_{}.shp'.format(weighing, current_country,nuts_class)))
        #pref_routes.to_pickle( (country_dir / 'optimal_routes_{}_{}_{}.pickle'.format(weighing, current_country,nuts_class)))
        logger.info("Optimal routes of {} saved to {}".format(current_country, country_dir))
        # Save optimal routes as feather to load quickly in the percolation_optimized.py
        pref_routes_df = pd.DataFrame(pref_routes[['o_node', 'd_node', 'origin', 'destination', 'v_ids', weighing, 'e_ids']])
        if current_country == 'norway':
            pref_routes_df.reset_index(drop=True,inplace=True)
        pref_routes_df.to_feather(od_matrix)

        #To enable inspection:
        # Save the edges and nodes of the graph
        gdf_to_shp(gpd.GeoDataFrame(nodes, geometry='geometry', crs='epsg:4326'),
                   (country_dir / 'inferred_nodes_{}.shp'.format(current_country)))
        # gdf_to_shp(gpd.GeoDataFrame(network, geometry='geometry', crs='epsg:3035'),
        #            os.path.join(output_folder.format(current_country), 'edges_{}.shp'.format(current_country)))

        G.write_pickle(fname=(country_dir / '{}_G.pkl'.format(current_country)))


    except Exception as Argument:
        logger.exception(str(Argument))


#Test list combinations
def old_list_combinations(max_aoi):
    list_combinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                         1000, 1100, 1200, 1300, 1400, 1500]
    list_combinations = [x for x in list_combinations if x < max_aoi]
    list_combinations.append(max_aoi)
    list_combinations = [str(x) for x in list_combinations]
    return list_combinations

def new_list_combinations(max_aoi):
    from math import floor

    list_combinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                         1000, 1100, 1200, 1300, 1400, 1500,1600,1700]
    list_combinations.append(max_aoi)
    if max_aoi <= 250:
        twenties = list(range(0,201,20))
        list_combinations.extend(twenties)

    if 200 < max_aoi <= 600:
        fifties = list(range(0,601,50))
        list_combinations.extend(fifties)
        list_combinations.append(max_aoi-10)

    elif max_aoi > 600:
        last_sampled = floor(max_aoi/100)*100 #last manyfold of hundred that was sampled
        delta = max_aoi-last_sampled
        if delta > 50:
            list_combinations.append(last_sampled+50)
        else:
            list_combinations.append(last_sampled-50)
        list_combinations.append(max_aoi-20)

    #Give a smooth end to the curves
    list_combinations.extend([max_aoi-2,max_aoi-5])

    list_combinations = [x for x in list_combinations if x <= max_aoi]
    list_combinations = sorted(list(set(list_combinations)))
    list_combinations = [str(x) for x in list_combinations]
    return list_combinations


if False: #run this block to do create the centroids file
    #3/5/2021: made new versions of Frederiques initital centroids files, because some NUTS-3 regions seemed to be missing
    # in Latvia and Sweden (among others)
    config = load_config()
    nuts_2_regions = config['paths']['NUTS_shapes'] / 'NUTS_RG_01M_2016_3035_LEVL_2.shp'
    nuts_3_regions = config['paths']['NUTS_shapes'] / 'NUTS_RG_01M_2016_3035_LEVL_3.shp'

    assert nuts_2_regions.exists()
    assert nuts_3_regions.exists()

    out_file_n2 = config['paths']['data'] / 'europe_nuts2_centroids.feather'
    out_file_n3 = config['paths']['data'] / 'europe_nuts3_centroids.feather'

    assert (not out_file_n2.exists())
    #assert (not out_file_n3.exists())

    df = create_centroids_csv(nuts_2_regions, ['LEVL_CODE', 'FID'],out_file_n2)

    print('end')



if __name__ == '__main__':
    from random import shuffle
    #countries = ['ALB', 'AUT', 'BEL', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'EST', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC',
    #            'HRV', 'HUN', 'IRL', 'ITA', 'LTU', 'LVA', 'MKD', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB', 'SVK','SVN', 'SWE']

    countries = ['NOR']
    nuts_class = 'nuts3'
    print(countries)

    #Single run
    optimal_routes(countries[0],nuts_class=nuts_class,weighing='time',
                   config_file='config.json',special_setting='Norway_relocate')

    #Multiple runs (sequential)

    for country in countries:
        try:
            if country in ['BEL','NLD','GBR','ITA','DEU']:
                nuts_class = 'nuts2'
            else:
                nuts_class = 'nuts3'
            optimal_routes(country,nuts_class=nuts_class,weighing='time',config_file='config.json',special_setting=None)
        except Exception as e:
            print(country,e)

    #Parallel processing
    #with Pool(4) as pool:
    #    pool.map(optimal_routes, countries, chunksize=1)

