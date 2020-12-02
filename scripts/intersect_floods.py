# -*- coding: utf-8 -*-
"""
Created on 2-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import os, sys

sys.path.append(r"D:\COACCH_paper\trails-master\src\trails")
# folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(folder)
# os.chdir(os.path.dirname(folder))  # set working directory to top folder

import pygeos
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from flow_model import create_graph
import igraph as ig
import rasterio
from statistics import mean, mode
from shapely import wkt
from shapely.ops import transform

# Location of the JRC floodmaps
floodmap_RP100_path = r"D:\COACCH_paper\data\JRC_floodmap_local\floodmap_EFAS_RP100_3857.tif"
aoimap_RP100_path = r"D:\COACCH_paper\data\JRC_floodmap_local\AreaOfInfl_EFAS_RP100y_3857.tif"

# Location of graphs of all countries in Europe, saved in *.feather format
networks_europe_path = r'D:\COACCH_paper\data\networks_europe_elco_koks'
network_files = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path)]

# Get the country codes from the *.feather files (networks) and see with a country code
# translation table which countries are there
country_codes = [f.split('-nodes')[0] for f in os.listdir(networks_europe_path) if f.endswith('-nodes.feather')]
translate_cntr_codes = pd.read_csv(r".\europe_flood_road_disruption\data\country_codes.csv", delimiter=';').set_index('code').to_dict(orient='dict')
print(len(country_codes), 'countries:\n' + '\n'.join([translate_cntr_codes['country'][cntry] for cntry in country_codes]))


i = 0  # iterate over the graphs
current_country = translate_cntr_codes['country'][network_files[i].split('-')[0].split('\\')[-1]]
print(current_country)
network = pd.read_feather(network_files[i])
network['geoms'] = pygeos.io.to_wkt(pygeos.from_wkb(network.geometry))
network['geoms'] = network['geoms'].apply(wkt.loads)

# lon and lat are swapped: swap them how they should be
# network['geoms'] = network.geoms.map(lambda line: transform(lambda x, y: (y, x), line))

network.drop('geometry', axis=1, inplace=True)
network.rename(columns={'geoms': 'geometry'}, inplace=True)

# Create a GeoDataFrame from the pandas DataFrame with the CRS EPSG:4326
network_geoms = gpd.GeoDataFrame(network, geometry='geometry', crs='EPSG:4326')
network_geoms = network_geoms.to_crs('EPSG:3857')

# plot the road network of the country that is being intersected
network_geoms.plot()
plt.title(current_country)
plt.show()


def hazard_intersect_gdf(gdf, hazard, hazard_name, res=1, agg='max'):
    """adds hazard values (flood/earthquake/etc) to the roads in a geodataframe
    Args:
        gdf [geopandas geodataframe]
        hazard [string]: full path to hazard data
        agg [string]: choose from max, min or mean; when multiple sections of a road
            are overlapping, aggregate the data in this way
    Returns:
        Graph with the added hazard data, when there is no hazard, the values is 0
    """
    # Create a new column in the GeoDataFrame with the name of the hazard attribute
    gdf[hazard_name] = 0

    # import and append the hazard data
    if hazard.endswith('.tif'):
        # GeoTIFF
        src = rasterio.open(hazard)
        print("Raster projection:", src.crs)
        # check which road is overlapping with the flood and append the flood depth to the graph
        for row in range(len(gdf)):
            if gdf.iloc[row]['geometry']:
                # check how long the road stretch is and make a point every other meter
                if res == 1:
                    nr_points = round(gdf.iloc[row]['geometry'].length)
                else:
                    nr_points = round(gdf.iloc[row]['geometry'].length / 50)
                if nr_points == 1:
                    coords_to_check = list(gdf.iloc[row]['geometry'].boundary)
                else:
                    coords_to_check = [gdf.iloc[row]['geometry'].interpolate(i / float(nr_points - 1), normalized=True) for i in
                                       range(nr_points)]
                crds = []
                for c in coords_to_check:
                    # check if part of the linestring is inside the flood extent
                    if (src.bounds.left < c.coords[0][0] < src.bounds.right) and (src.bounds.bottom < c.coords[0][1] < src.bounds.top):
                        crds.append(c.coords[0])
                if crds:
                    values_list = [x.item(0) for x in src.sample(crds)]
                    # the road lays inside the flood extent
                    if agg == 'max':
                        if (max(values_list) > 999999) | (max(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.iloc[row][hazard_name] = 0
                        else:
                            gdf.iloc[row][hazard_name] = max(values_list)
                    elif agg == 'min':
                        if (min(values_list) > 999999) | (min(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.iloc[row][hazard_name] = 0
                        else:
                            gdf.iloc[row][hazard_name] = min(values_list)
                    elif agg == 'mean':
                        if (mean(values_list) > 999999) | (mean(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.iloc[row][hazard_name] = 0
                        else:
                            gdf.iloc[row][hazard_name] = mean(values_list)
                    elif agg == 'mode':
                        if (mode(values_list) > 999999) | (mode(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.iloc[row][hazard_name] = 0
                        else:
                            gdf.iloc[row][hazard_name] = mode(values_list)
                    else:
                        print("No aggregation method is chosen ('max', 'min', 'mean' or 'mode').")

    elif hazard.endswith('.shp'):
        print("Shapefile overlay not yet implemented")
        # # Shapefile
        # gdf = gpd.read_file(hazard)
        # spatial_index = gdf.sindex
        #
        # for u, v, k, edata in graph.edges.data(keys=True):
        #     if 'geometry' in edata:
        #         possible_matches_index = list(spatial_index.intersection(edata['geometry'].bounds))
        #         possible_matches = gdf.iloc[possible_matches_index]
        #         precise_matches = possible_matches[possible_matches.intersects(edata['geometry'])]
        #
        #         if not precise_matches.empty:
        #             if agg == 'max':
        #                 graph[u][v][k][hazard_name] = precise_matches[hazard_name].max()
        #             if agg == 'min':
        #                 graph[u][v][k][hazard_name] = precise_matches[hazard_name].min()
        #             if agg == 'mean':
        #                 graph[u][v][k][hazard_name] = precise_matches[hazard_name].mean()
        #         else:
        #             graph[u][v][k][hazard_name] = 0
        #     else:
        #         graph[u][v][k][hazard_name] = 0

    return gdf


# Overlay the geodataframe of the network with the (flood) hazard for the flood depth
network_flood = hazard_intersect_gdf(network_geoms, floodmap_RP100_path, 'floodDepth', res=50, agg='max')

# Overlay the geodataframe of the network with the (flood) hazard event data for the "area of influence" (event)
network_flood = hazard_intersect_gdf(network_flood, aoimap_RP100_path, 'AoI_RP100', res=50, agg='mode')

network_flood['floodDepth'].max()

# plot the flood depth on the roads in the network
network_flood.plot(column='floodDepth')
plt.title(current_country)
plt.show()

graph = create_graph(network_flood)[0]
print(ig.summary(graph))

list(graph.es())[0]
