# -*- coding: utf-8 -*-
"""
Created on 15-12-2020

@author: Frederique de Groen & Kees van Ginkel

Part of a COACCH percolation analysis of European road networks.

"""

import os, sys

sys.path.append(r"P:\osm_flood\network_analysis\igraph\trails\src\trails")

# folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(folder)
# os.chdir(os.path.dirname(folder))  # set working directory to top folder

import igraph as ig
import pygeos as pyg
import pandas as pd
import geopandas as gpd
from network import nearest
import itertools
import feather
from shapely import wkt
from shapely.geometry import MultiLineString
import numpy as np
from numpy import object as np_object
from tqdm import tqdm
from pathos.multiprocessing import Pool


# NUTS-3 regions of Europe
# nuts_3_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_3.shp"
# nuts_2_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_2.shp"

# list of islands that aren't assessed
overseas = ['PT200', 'PT300',  # Portugal: Azores and Madeira
            'ES531', 'ES532', 'ES533', 'ES703', 'ES704', 'ES705', 'ES706', 'ES707', 'ES708', 'ES709', 'ES630', 'ES640',  # Spain: Canary Islands
            'FRY10', 'FRY20', 'FRY30', 'FRY40',  # France: overseas areas: Gouadeloupe, Martinique, French Guiana, La Reunion, Mayotte "])
            'FRY50', 'FRM01', 'FRM02',
            'EL307', 'EL411', 'EL412', 'EL413', 'EL421', 'EL422', 'EL431', 'EL432', 'EL433', 'EL434', 'EL621', 'EL622', 'EL623',  # Greece
            'HR037',  # croatia
            'UKN0',  # UK (NUTS-2 classification)
            'DK014',  # Denmark
            'FI200',  # Finland
            'SE214'  # Sweden
             ]

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


# create_centroids_csv(nuts_2_regions, ['LEVL_CODE', 'FID'], os.path.join(input_folder, 'europe_nuts2_centroids.feather'))


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


# iterate over the graphs
#todo: paths should use the config file!!!
#Todo: NuTS2/NUTS3 should not be hardcoded, but given to the function
def optimal_routes(cntry):
    """
    Preprocessing: finds the optimal routes between the NUTS-3 or NUTS-2 regions in a country

    :param cntry:
    :return:

    Requires
     - feather files with edges and nodes of the road network of the country
     - feather file with the centroids of the NUTS2 or NUTS-3 regions in the country
    """


    # define in- and output folders
    input_folder = r"P:\osm_flood\network_analysis\data"
    output_folder = r"P:\osm_flood\network_analysis\data\output\{}"

    # Location of graphs of all countries in Europe, saved in *.feather format
    networks_europe_path = os.path.join(input_folder, 'networks_intersect_hazard_elco_koks')
    edge_file = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if
                 f == cntry + '-edges.feather'][0]

    # Get the country codes from the *.feather files (networks) and see with a country code
    # translation table which countries are there
    translate_cntr_codes = pd.read_csv(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\data\country_codes.csv",
                                       delimiter=';').set_index('code3').to_dict(orient='dict')

    # set the weighing (time or distance)
    weighing = 'time'

    # cntry_code = network_files[i].split('-')[0].split('\\')[-1]
    current_country = translate_cntr_codes['country'][cntry].lower()
    print("\nCurrent iteration is for:", current_country)

    od_matrix = os.path.join(output_folder.format(current_country),
                 'optimal_routes_{}_{}.feather'.format(weighing, current_country))

    # create the folder if it does not exist yet
    if not os.path.exists(output_folder.format(current_country)):
        os.makedirs(output_folder.format(current_country))

    if os.path.exists(od_matrix):
        print('country = {} already finished!'.format(current_country))
        return None

    # The centroids of the NUTS-3 regions
    # TODO: change this to NUTS-2 centroids if necessary
    centroids = pd.read_feather(r"P:\osm_flood\network_analysis\data\europe_nuts3_centroids.feather")

    # select the centroids that are in the country that is analysed
    # discard the centroids of the small islands that are not reachable from mainland Europe
    selected_centroids = centroids.loc[(centroids['CNTR_CODE'] == translate_cntr_codes['code2'][cntry]) &
                                       (~centroids['NUTS_ID'].isin(overseas))]

    try:
        selected_centroids['geometry'] = selected_centroids['geometry'].apply(pyg.from_wkt)

        # read the network files from Elco Koks
        network = pd.read_feather(edge_file)
        # network['geometry'] = network['geometry'].apply(wkt.loads)

        # create a geometry column with shapely geometries
        network['geoms'] = pyg.io.to_wkt(pyg.from_wkb(network.geometry))  # see if this should be activated with the new feather files
        network['geoms'] = network['geoms'].apply(wkt.loads)
        network.drop('geometry', axis=1, inplace=True)
        network.rename(columns={'geoms': 'geometry'}, inplace=True)

        # todo: filter out the tertiary roads?
        # network_geoms = network_geoms.loc[~network_geoms['highway'].isin(['tertiary', 'tertiary_link'])]

        # create the graph
        G = graph_load(network, ['geometry', 'id', 'RP100_cells_intersect', 'RP100_max_flood_depth',
                                 'AoI_RP100y_majority', 'AoI_RP100y_unique', 'fds_majority', 'fds__unique'])

        # read nodes
        nodes = feather.read_dataframe(edge_file.replace("-edges", "-nodes"))
        nodes.geometry = pyg.from_wkb(nodes.geometry)

        # TODO: Change to NUTS-2 if necessary
        nuts_class = 'nuts3'

        # find the nodes that are closest to the centroids of the NUTS-3 regions
        node_ids_od_pairs = prepare_possible_OD_EU(selected_centroids, nodes, tolerance=0.1)
        for n_id, nuts3_code in node_ids_od_pairs:
            nodes.loc[nodes['id'] == n_id, nuts_class] = nuts3_code  # Not per se nuts3, can also be nuts2

        # Add the nodes to the graph
        G.add_vertices(len(nodes))
        G.vs['id'] = nodes['id']
        G.vs[nuts_class] = nodes[nuts_class]

        print(G.summary())

        # Save nodes as feather
        nodes.geometry = pyg.to_wkb(nodes.geometry)
        nodes.to_feather(edge_file.replace("-edges", "-nodes_{}".format(nuts_class)))

        # save the list of combinations that should be ran in the excel sheet that is also used to check whether
        # the number of optimal routes is correct
        # TODO: save indeed to excel to check if the number of optimal routes is correct (now only the list of combinations is written to the csv)
        all_aois = list(set([int(x) for l in list(network['AoI_RP100y_unique']) for x in l if (x != 0) and (x == x)]))
        max_aoi = len(all_aois)

        list_combinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                             1000, 1100, 1200, 1300, 1400, 1500]
        list_combinations = [x for x in list_combinations if x < max_aoi]
        list_combinations.append(max_aoi)
        list_combinations = [str(x) for x in list_combinations]

        combinations_csv = r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\data\nuts3_combinations_completeness.csv"
        df = pd.read_csv(combinations_csv)
        nr_optimal_routes = df.loc[df['code3'] == cntry, 'nr_routes'].iloc[0]
        df.loc[df['code3'] == cntry, 'aoi_combinations'] = " ".join(list_combinations)
        df.to_csv(combinations_csv)

        # dataframe to save the optimal routes
        pref_routes = gpd.GeoDataFrame(columns=['o_node', 'd_node', 'origin', 'destination', 'AoIs',
                                                'v_ids', weighing, 'e_ids', 'geometry'],
                                       geometry='geometry', crs='epsg:3035')

        # create the routes between all OD pairs
        for o, d in tqdm(itertools.combinations(node_ids_od_pairs, 2), desc='NUTS-3 optimal routes {}'.format(current_country)):
            # calculate the length of the optimal route
            # now the graph is treated as directed, make mode=ig.ALL to treat as undirected
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
            print("The number of preferred routes does not match with the number it should be. Check for country", cntry)

        # remove impossible routes (with time = inf)
        pref_routes = pref_routes.replace([np.inf, -np.inf], np.nan).dropna(subset=[weighing])

        # Save optimal routes as shapefile
        gdf_to_shp(pref_routes, os.path.join(output_folder.format(current_country), 'optimal_routes_{}_{}.shp'.format(weighing, current_country)))
        print("Optimal routes of {} saved to {}".format(current_country,
                                                        os.path.join(output_folder.format(current_country), 'optimal_routes_{}_{}.shp'.format(weighing, current_country))))

        # Save optimal routes as feather to load quickly in the percolation_optimized.py
        pref_routes_df = pd.DataFrame(pref_routes[['o_node', 'd_node', 'origin', 'destination', 'v_ids', weighing, 'e_ids']])
        pref_routes_df.to_feather(od_matrix)

        # Save the edges and nodes of the graph
        # gdf_to_shp(gpd.GeoDataFrame(nodes, geometry='geometry', crs='epsg:4326'),
        #            os.path.join(output_folder.format(current_country), 'nodes_{}.shp'.format(current_country)))
        # gdf_to_shp(gpd.GeoDataFrame(network, geometry='geometry', crs='epsg:3035'),
        #            os.path.join(output_folder.format(current_country), 'edges_{}.shp'.format(current_country)))
    except KeyError as e:
        print(current_country, 'is not an EU memberstate.', e)


if __name__ == '__main__':
    # countries = ['ALB', 'AUT', 'BEL', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GIB', 'GRC',
    #              'HRV', 'HUN', 'IRL', 'ITA', 'LUX', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB', 'SVK', 'SWE']
    #countries = ['EST', 'LTU', 'LVA', 'MKD','SVN','SWE','DNK']
    countries = ['SVN']
    from random import shuffle
    shuffle(countries)
    print(countries)

    optimal_routes(countries[0])

    #with Pool(4) as pool:
    #    pool.map(optimal_routes, countries, chunksize=1)
