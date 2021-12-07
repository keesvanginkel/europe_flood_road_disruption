

from Europe_utils import *
from pathlib import Path
import json
from percolation_optimized_parallel import import_graph,import_optimal_routes
import time
import pygeos as pyg
import igraph as ig
from tqdm import tqdm
import numpy as np
from statistics import mean
import geopandas as gpd
from copy import copy, deepcopy


from shapely.geometry import MultiLineString


weighing = 'time'

def graph_load_v2(edges,save_columns='all'):
    """Creates a graph from an edges dataframe

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns
        save_columns (list of str): a list of the names of the columns that should be added to the graph

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """
    #return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)
    graph.vs['id'] = graph.vs['name']

    return graph

def graph_load_v3(edges):
    """Creates a graph from an edges dataframe

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns
        save_columns (list of str): a list of the names of the columns that should be added to the graph

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes


    This should mimic the version that Frederique used, but with some extra metadata
    """
    save_columns =    ['osm_id',
     'geometry',
     'highway',
     'oneway',
     'lanes',
     'maxspeed',
     'id',
     'distance',
     'time',
     'RP020_cells_intersect',
     'RP020_min_flood_depth',
     'RP020_max_flood_depth',
     'RP020_mean_flood_depth',
     'RP500_cells_intersect',
     'RP500_min_flood_depth',
     'RP500_max_flood_depth',
     'RP500_mean_flood_depth',
     'RP200_cells_intersect',
     'RP200_min_flood_depth',
     'RP200_max_flood_depth',
     'RP200_mean_flood_depth',
     'RP050_cells_intersect',
     'RP050_min_flood_depth',
     'RP050_max_flood_depth',
     'RP050_mean_flood_depth',
     'RP010_cells_intersect',
     'RP010_min_flood_depth',
     'RP010_max_flood_depth',
     'RP010_mean_flood_depth',
     'RP100_cells_intersect',
     'RP100_min_flood_depth',
     'RP100_max_flood_depth',
     'RP100_mean_flood_depth',
     'AoI_RP500y_majority',
     'AoI_RP500y_unique',
     'AoI_RP50y_majority',
     'AoI_RP50y_unique',
     'AoI_RP100y_majority',
     'AoI_RP100y_unique',
     'AoI_RP200y_majority',
     'AoI_RP200y_unique',
     'AoI_RP20y_majority',
     'AoI_RP20y_unique',
     'AoI_RP10y_majority',
     'AoI_RP10y_unique',
     'fds_majority',
     'fds__unique']


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

def import_graph_v2(country_code, nuts_class='nuts3', config_file='config.json'):
    """
    Arguments:
        *countr_code* (string) : 3-letter code of country name e.g. 'Bel'
        *nuts_class* (string) : 'nuts3' or 'nut2'
        *config_file* (string) : name of the config file directing to the path, default = config.json

    Returns:
        **

    """
    config = load_config(file=config_file)
    networks_path = config['paths']['graphs_folder']

    edge_file = networks_path / '{}-edges.feather'.format(country_code)
    assert edge_file.exists()

    # read the network files from Elco Koks, the edges files already contain the flood hazard data
    edges = pd.read_feather(edge_file)
    edges.geometry = pyg.from_wkb(edges.geometry)

    # create the graph, this can in principle be done from edges alone
    G = graph_load_v2(edges)

    # load node file that has the nuts-origin-destinations (created in preprocessing)
    nodes = pd.read_feather(networks_path / '{}-nodes_{}.feather'.format(country_code,nuts_class))

    nodes.geometry = pyg.from_wkb(nodes.geometry)
    #Todo: only add the new nodes. For now, leave this as it is.
    #extra_nodes = nodes.loc[~nodes['nuts2'].isnull()]

    # Add the nodes to the graph
    G.add_vertices(len(nodes))
    G.vs['id'] = nodes['id']
    G.vs[nuts_class] = nodes[nuts_class]

    # print(G.summary())
    return G

def import_graph_v3(country_code, nuts_class='nuts3', config_file='config.json'):
    """
    Arguments:
        *countr_code* (string) : 3-letter code of country name e.g. 'Bel'
        *nuts_class* (string) : 'nuts3' or 'nut2'
        *config_file* (string) : name of the config file directing to the path, default = config.json

    Returns:
        **

    """
    config = load_config(file=config_file)
    networks_path = config['paths']['graphs_folder']

    edge_file = networks_path / '{}-edges.feather'.format(country_code)
    assert edge_file.exists()

    # read the network files from Elco Koks, the edges files already contain the flood hazard data
    edges = pd.read_feather(edge_file)
    edges.geometry = pyg.from_wkb(edges.geometry)

    # create the graph, this can in principle be done from edges alone
    G = graph_load_v3(edges)

    # load node file that has the nuts-origin-destinations (created in preprocessing)
    nodes = pd.read_feather(networks_path / '{}-nodes_{}.feather'.format(country_code,nuts_class))
    nodes.geometry = pyg.from_wkb(nodes.geometry)
    #Todo: only add the new nodes. For now, leave this as it is.
    #extra_nodes = nodes.loc[~nodes['nuts2'].isnull()]

    # Add the nodes to the graph
    G.add_vertices(len(nodes))
    G.vs['id'] = nodes['id']
    G.vs[nuts_class] = nodes[nuts_class]

    # print(G.summary())
    return G

### The function below is based on run_percolation_parallel.py -> stochastic_network_analysis_phase2()
def evaluate_event(path_to_event_json,config_file,country_code,nuts_class):
    """
    Calculate metrics for a single flood event, prescribed in an event json.

    Currently (8/11/2021), these jsons are directly prepared in a notebook, no need to run prep_par

    Arguments:
        *G* (iGraph graph) : copy of the undisturbed graph object
        *path_to_event_json* (pathlib path object) : path to json describing the events
        *config_file* (string) : name of the config file in the main directory
        *country_code* (string) : 3l country code e.g. 'DEU'
        *nuts_class* (string) : 'nuts3' or 'nuts2'

    Returns:
        None

    Effect:
        write the results of the analysis to a csv file in the folder
            config['paths']['main_output] / *country name* / finished

    """
    t0 = time.time()

    #some new settings:
    removed_to_shape = True #saves the removed edges to config['paths']['main_output] / *country name* / checkpoints
    new_routes_to_shape = True #save the new routes to shape

    config = load_config(file=config_file)

    #LOAD AND UNPACK DATA FROM JSON FILE
    with open(path_to_event_json) as f:
        event = json.load(f)
    event_data = event['data']
    year = list(event_data.keys())[0]

    #CHECK IF PROCESS IS ALREADY RUN:
    country_name = country_names(country_code)
    result_path = config['paths']['main_output'] / country_name.lower() / 'finished' / '{}.csv'.format(year)

    if result_path.exists():
        print('{} already finished'.format(year))
        return None

    #LOAD ORIGINAL OD-MATRIX
    od_optimal_routes = import_optimal_routes(country_name, config_file=config_file)

    #t1 = time.time()
    #print('Time for first part of function:',t1-t0)

    # LOAD THE NETWORK
    ### TODO: PROBABLY MUCH FASTER TO LOAD THE NETWORK ONCE OUTSIDE THE FUNCTION, AND GIVE A COPY TO THE FUNCTION!!!
    G = import_graph_v3(country_code, nuts_class=nuts_class, config_file=config_file)
    #t2 = time.time()
    #print('Time for loading graph', t2 - t1)

    # initiate result metric variables
    df = pd.DataFrame(columns=['year', 'disrupted', 'avg extra time', 'no detour'])
    tot_routes = len(od_optimal_routes.index)

    #Iterate over the different basins
    for basin_flood in event_data[year]['microfloods']:
        basin_aoi = basin_flood['basin_aoi']
        return_period = basin_flood['return_period']
        cell_aois = basin_flood['cell_aois']

        if not isinstance(cell_aois,list):
            raise TypeError('Unexpected type for cell_aois; should be list  but is:',cell_aois,type(cell_aois))

        if len(cell_aois) == 0:
            continue
        else:

            #Select the aoi and flood raster that best represent the flood of this return period
            aoi_col, depth_col = find_closest_raster(return_period)
            assert aoi_col in G.es.attributes()
            assert depth_col in G.es.attributes()

            #Complicated function call to avoid forloop (very slow)
            # - compare intersect between set of cell aois, with for each edge: the aoi that overlap with that edge
            # - and also check if the return period of the flood is larger than the flood protection of this edge
            to_remove = G.es.select(lambda e: (set(cell_aois) & set(e[aoi_col])) and (return_period > e['fds_majority']))
            #print(basin_aoi,len(to_remove))

            #save all removed edges in a dataframe to inspect in GIS
            if removed_to_shape:
                df_to_remove = ES_to_df(to_remove)
                #print(len(df_to_remove))
                if not 'df_all_removed' in locals(): #first time that edges are removed
                    df_all_removed = df_to_remove.copy()
                else:
                    df_all_removed = df_all_removed.append(df_to_remove)

            G.delete_edges(to_remove)
            #print(len(G.es))

    t3 = time.time()
    #print('Time for iterating over floods per basin', t3 - t2)

    if removed_to_shape:
        check_path = config['paths']['main_output'] / country_name.lower() / 'checkpoints'
        if not check_path.exists() : check_path.mkdir()
        gdf = gpd.GeoDataFrame(df_all_removed,crs='EPSG:3035')
        gdf.to_file((check_path / '{}_removed_edges.shp'.format(year)), sep=';')

        t3b = time.time()
        #print('Time for saving shapefile with removed edges', t3b - t3)
        t3 = time.time()



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
        # G.get_shortest_paths(v=int(o), to=int(d), mode=ig.OUT, weights=weighing,output='vpath')

        alt_route = alt_route[0][0]
        if alt_route != np.inf:
            # alt_route = inf if the route is not available
            # append to list of alternative routes to get the average
            extra_time.append(alt_route - od_optimal_routes.iloc[ii][weighing])  # changed 'time' into weighing
            if od_optimal_routes.iloc[ii][weighing] != alt_route:  # changed 'time' into weighing
                # the alternative route is different from the preferred route
                disrupted += 1
                if True: #optionally, save the alternative paths for this disruption
                    alt_route = calc_alt_route(G,o,d,weighing,od_optimal_routes)
                    if not 'all_alt_routes' in locals():
                        all_alt_routes = alt_route.copy()
                    else:
                        all_alt_routes = all_alt_routes.append(alt_route)
        else:
            # append to calculation dataframe
            disrupted += 1
            nr_no_detour += 1

    #t4 = time.time()
    #print('Time for calculing new routes', t4 - t3)

    output = {'year': year,
              'disrupted': disrupted / tot_routes * 100,
              'avg extra time': mean(extra_time),
              'no detour': nr_no_detour / tot_routes * 100}
    df = df.append(output, ignore_index=True)
    df.to_csv(result_path, sep=';')

    if new_routes_to_shape:
        check_path = config['paths']['main_output'] / country_name.lower() / 'checkpoints'
        if not check_path.exists(): check_path.mkdir()
        gdf = gpd.GeoDataFrame(all_alt_routes, crs='EPSG:3035')
        gdf.to_file((check_path / '{}_alt_routes.shp'.format(year)), sep=';')

    t5 = time.time()
    print('total time:',t5-t0)





            #todo: remove below code if the new selection function works well.
            #for aoi in cell_aois:
                #https://igraph.org/python/doc/api/igraph.VertexSeq.html#select
                #cands = G.es.select(lambda e: aoi in e[aoi_col])
                #print(aoi,len_cands)
                #This avoids the forloop
                #cands = G.es.select(lambda e: (aoi in e[aoi_col]) and (return_period > e['fds_majority']))
                #for c in cands: #iterate over edges that are candidate for removing (because they are intersected by aoi raster)
                #    fpl = c['fds_majority'] #flood protection at this edge
                #    if return_period > fpl: #flood RP exceeds flood protection at this edge
                #        #print('add candidate',c,'to to_remove')
                #        to_remove.append(c)


    return None






def find_closest_raster(return_period,aoi_col='AoI_RP{}y_unique',depth_col='RP{}_max_flood_depth'):
    """
    Find the closest AoI and Flood raster column name for given return period

    Arguments:
        *return_period* (float): Return period of the flood for which to find the nearest inundation raster
        *aoi_col* (str): the format of the column name to find, default can be changed to anything in G.es.attributes
        *depth_col* (str): the format of the column name to find, default can be changed in G.es.attributes

    Returns:
        *aoi_col* (str):  e.g. 'AoI_RP10y_majority'
        *depth_col* (str) : e.g. 'RP500_max_flood_depth'

    """
    assert return_period > 0
    available_rps = [10,20,50,100,200,500]
    nearest = min(available_rps, key=lambda x:abs(x-return_period))

    #note the difference in notation: AoI: 'RP10...', flood: 'RP010...'
    aoi_col = aoi_col.format(nearest)
    if len(str(nearest)) == 2: # e.g. RP10 -> RP010
        depth_col = depth_col.format('0'+str(nearest))
    elif len(str(nearest)) == 3: # e.g. RP200 -> RP200
        depth_col = depth_col.format(nearest)
    else:
        raise ValueError('Does not know how to handle value nearest = {}, valid are e.g. 10, 500'.format(nearest))

    return aoi_col, depth_col

def calc_alt_route(G,o,d,weighing,odf):
    """
    Function that calculates the path over the network, in addition to the travel times that are the default output.
        *G* (graph)
        *o* (origin)
        *d* (destination)
        *odf* = od_optimal_routes dataframe

    Returns:
        *df* : dataframe with route data and geometries

    """
    path_edges = G.get_shortest_paths(v=int(o), to=int(0), weights=weighing, mode=ig.OUT, output="epath")
    edges_geoms = pyg.multilinestrings(G.es[path_edges[0]]['geometry'])
    path_edges_ids = G.es[path_edges[0]]['id']

    #lookup the corresponding nutsnames of orign and destination
    od_pair = odf.loc[(odf['o_node'] == o) & (odf['d_node'] == d)]

    data = {'o_node': o, 'd_node': d, 'origin': od_pair['origin'],
            'destination': od_pair['destination'], 'e_ids': str(path_edges_ids), 'geometry': edges_geoms}
    df = pd.DataFrame(data)
    return df


def ES_as_csv(es,path):
    """"
    Args:
        *es* (G.es) : EdgeSequence
        *path* (Path) : pathlib path to save the shapefile to
    
    Returns: 
        None
        
    Effect: saves the simplified edgesequence as a shapefile.
    """
    attributes = list(es.attributes())
    keep_attributes = ['distance', 'time', 'osm_id', 'geometry', 'highway', 'id']
    df = pd.DataFrame(columns=keep_attributes)
    for attribute in keep_attributes:
        df[attribute] = es.get_attribute_values(attribute)
    df.to_csv(path,sep='')


def ES_to_df(es,
    keep_attributes = ['distance', 'time', 'osm_id', 'geometry', 'highway', 'id']):
    """"
    Save Igraph EdgeSequence to dataframe while keeping some of the attribute data

    Args:
        *es* (G.es) : EdgeSequence
        *keep_attributes* (list) : list of strings of the attributes name you want to keep
    #to see all attributes:
    #attributes = list(es.attributes())

    Returns: 
        df

    """
    df = pd.DataFrame(columns=keep_attributes)
    for attribute in keep_attributes:
        df[attribute] = es.get_attribute_values(attribute)
    return df

### TODO: IMPLEMENT THIS FASTER WAY OF CALCULATION
# def run_shortest_paths(graph, OD_nodes, weighting='time'):
#     collect_all_values = []
#     for x in (range(len(OD_nodes))):
#         get_paths = graph.get_shortest_paths(OD_nodes[x], OD_nodes, weights=weighting, output='epath')
#
#         collect_value = []
#         for path in get_paths:
#             if len(path) == 0:
#                 collect_value.append(0.0)
#             else:
#                 collect_value.append(sum(graph.es[path]['time']))
#
#         collect_all_values.append(collect_value)
#
#     return np.matrix(collect_all_values)


def call_evaluate_event(tuple):
    """
    Helper function for parallel processing.

    Unpacks the tuple and gives args to evaluate_event

    """
    path_to_event_json = tuple[0]
    config_file = tuple[1]
    country_code =  tuple[2]
    nuts_class = tuple[3]
    print('Preparing to run {}'.format(path_to_event_json.stem))
    evaluate_event(path_to_event_json=path_to_event_json,
                   config_file=config_file,
                   country_code=country_code,
                   nuts_class=nuts_class)
    print('Function evaluate_event finished for {}'.format(path_to_event_json.stem))
    return None


if __name__ == '__main__':
    print('Running a test of the function evaluate_event')
    country_code = 'DEU'
    nuts_class = 'nuts2'

    sample_file = Path('D:\Europe_percolation\event_sampling_vs2\main_output\germany\scheduled\year_87.json') #87
    config_file = 'config_eventbased_2.json'


    #New parallel processing set-up (first load graph)
    #This does not work, because TypeError: cannot pickle 'pygeos.lib.GEOSGeometry' object
    #print('start loading graph')
    #G = import_graph_v3(country_code, nuts_class=nuts_class, config_file=config_file)
    #print('finished loading graph')

    print('start evaluating events')
    #linear call of the function
    evaluate_event(path_to_event_json=sample_file,config_file=config_file,country_code=country_code,nuts_class=nuts_class)



