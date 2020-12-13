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
import pandas as pd
import geopandas as gpd
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

# Location of the JRC floodmaps
floodmap_RP100_path = os.path.join(input_folder, r"JRC_floodmap_local\floodmap_EFAS_RP100_3857.tif")
aoimap_RP100_path = os.path.join(input_folder, r"JRC_floodmap_local\AreaOfInfl_EFAS_RP100y_3857.tif")

# Location of graphs of all countries in Europe, saved in *.feather format
networks_europe_path = os.path.join(input_folder, 'networks_europe_elco_koks')
network_files = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if f.endswith('-edges.feather')]

# Get the country codes from the *.feather files (networks) and see with a country code
# translation table which countries are there
country_codes = [f.split('-nodes')[0] for f in os.listdir(networks_europe_path) if f.endswith('-nodes.feather')]
translate_cntr_codes = pd.read_csv(r".\europe_flood_road_disruption\data\country_codes.csv", delimiter=';').set_index('code').to_dict(orient='dict')
print(len(country_codes), 'countries:\n' + '\n'.join([translate_cntr_codes['country'][cntry] for cntry in country_codes]))


def intersect_iterator(i):
    # iterate over the graphs
    current_country = translate_cntr_codes['country'][network_files[i].split('-')[0].split('\\')[-1]].lower()
    print("Current iteration is for:", current_country)

    # create the folder if it does not exist yet
    try:
        os.makedirs(output_folder.format(current_country))
    except OSError as e:
        print("Folder already exists:", e)

    # read the network files from Elco Koks
    network = pd.read_feather(network_files[i])

    # create a geometry column with shapely geometries
    network['geoms'] = pygeos.io.to_wkt(pygeos.from_wkb(network.geometry))
    network['geoms'] = network['geoms'].apply(wkt.loads)
    network.drop('geometry', axis=1, inplace=True)
    network.rename(columns={'geoms': 'geometry'}, inplace=True)

    # Create a GeoDataFrame from the pandas DataFrame with the CRS EPSG:4326
    network_geoms = gpd.GeoDataFrame(network, geometry='geometry', crs='EPSG:4326')
    network_geoms = network_geoms.to_crs('EPSG:3857')
    network_geoms = network_geoms.loc[~network_geoms['highway'].isin(['tertiary', 'tertiary_link'])]

    # clip the hazard map to the extent of the road network of the country that is being intersected with the hazard map
    def clip_raster(path_to_floodmap, country_bounds):
        xds = rioxarray.open_rasterio(path_to_floodmap, masked=True, chunks=True)
        to_clip = gpd.GeoDataFrame(geometry=[box(*country_bounds)], crs=xds.rio.crs.to_dict())
        return xds.rio.clip(to_clip.geometry.apply(mapping), to_clip.crs, drop=True)


    def hazard_intersect_gdf(gdf, hazard_xarray, hazard_name, res=1, agg='max'):
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
        print("Raster projection:", hazard_xarray.spatial_ref.crs_wkt)
        # check which road is overlapping with the flood and append the flood depth to the graph
        for row in range(len(gdf)):
            if gdf.iloc[row]['geometry']:
                # check how long the road stretch is and make a point every 'res' meters
                if res == 1:
                    nr_points = round(gdf.iloc[row]['geometry'].length)
                else:
                    nr_points = round(gdf.iloc[row]['geometry'].length / res)
                if nr_points == 1:
                    coords_to_check = list(gdf.iloc[row]['geometry'].boundary)
                else:
                    coords_to_check = [gdf.iloc[row]['geometry'].interpolate(i / float(nr_points - 1), normalized=True) for i in
                                       range(nr_points)]

                if coords_to_check:
                    crds = [c.coords[0] for c in coords_to_check]
                    values_list = [hazard_xarray.sel(x=xy[0], y=xy[1], method="nearest").values[0] for xy in crds]
                    # print(values_list)
                    # the road lays inside the flood extent
                    if agg == 'max':
                        if (np.nanmax(values_list) > 999999) | (np.nanmax(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.loc[row, hazard_name] = 0
                            # print('0')
                        else:
                            gdf.loc[row, hazard_name] = np.nanmax(values_list)
                            # print(np.nanmax(values_list))
                    elif agg == 'min':
                        if (np.nanmin(values_list) > 999999) | (np.nanmin(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.loc[row, hazard_name] = 0
                        else:
                            gdf.loc[row, hazard_name] = np.nanmin(values_list)
                    elif agg == 'mean':
                        if (np.nanmean(values_list) > 999999) | (np.nanmean(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.loc[row, hazard_name] = 0
                        else:
                            gdf.loc[row, hazard_name] = np.nanmean(values_list)
                    elif agg == 'mode':
                        # remove nan values because there is no such numpy.nanmode()
                        values_list = [item for item in values_list if not pd.isnull(item)]
                        if (mode(values_list) > 999999) | (mode(values_list) < -999999):
                            # the road is most probably in the 'no data' area of the raster (usually a very large or small number is used as 'no data' value)
                            gdf.loc[row, hazard_name] = 0
                        else:
                            gdf.loc[row, hazard_name] = mode(values_list)
                    else:
                        print("No aggregation method is chosen ('max', 'min', 'mean' or 'mode').")

        return gdf

    # overlay the flood depth map
    clipped = clip_raster(floodmap_RP100_path, network_geoms.total_bounds)

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    clipped.plot(ax=ax, cmap='winter')
    network_geoms.plot(ax=ax, edgecolor='black', linewidth=0.75)
    plt.title(current_country)
    # plt.show()
    plt.savefig(os.path.join(output_folder.format(current_country), current_country + "_roads_flood_depth.png"))
    plt.close()

    # Overlay the geodataframe of the network with the (flood) hazard for the flood depth
    network_flood = hazard_intersect_gdf(network_geoms, clipped, 'floodDepth', res=10, agg='max')
    print("Maximum flood depth:", network_flood['floodDepth'].max())

    # plot the flood depth on the roads in the network
    network_flood.plot(column='floodDepth')
    plt.title(current_country)
    # plt.show()
    plt.savefig(os.path.join(output_folder.format(current_country), current_country + "_flood_depth_on_roads.png"))
    plt.close()

    try:
        # overlay the Area of Influence map
        clipped = clip_raster(aoimap_RP100_path, network_flood.total_bounds)

        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        clipped.plot(ax=ax, cmap='spring')
        network_flood.plot(ax=ax, edgecolor='black', linewidth=0.75)
        plt.title(current_country)
        # plt.show()
        plt.savefig(os.path.join(output_folder.format(current_country), current_country + "_roads_aoi.png"))
        plt.close()

        # Overlay the geodataframe of the network with the (flood) hazard event data for the "area of influence" (event)
        network_flood = hazard_intersect_gdf(network_flood, aoimap_RP100_path, 'AoI_RP100', res=10, agg='mode')
    except Exception as e:
        print("failed because of", e)

    # save as shapefile to check the network
    network_flood.to_file(os.path.join(output_folder.format(current_country), current_country + "_network.shp"))

    # Create a graph from the geodataframe
    graph = create_graph(network_flood)[0]
    print(ig.summary(graph))

    gdf_edges = pd.DataFrame(list(graph.es['geometry']), columns=['geometry'])
    gdf_nodes = pd.DataFrame(list(graph.vs['geometry']), columns=['geometry'])

    gdf_edges.to_feather(os.path.join(output_folder.format(current_country), current_country + "_edges.feather"))
    gdf_nodes.to_feather(os.path.join(output_folder.format(current_country), current_country + "_nodes.feather"))

    print(current_country, "done!")


for ii in range(len(network_files)):
    intersect_iterator(ii)
