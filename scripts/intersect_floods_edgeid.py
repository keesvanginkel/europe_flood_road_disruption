import os
import pandas as pd
import numpy as np
import geopandas as gpd
import pygeos
from tqdm import tqdm
from rasterstats import zonal_stats
import pyproj
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool,cpu_count

def reproject(geometries):
    #Find crs of current df and arbitrary point(lat,lon) for new crs
    current_crs="epsg:4326"
    #The commented out crs does not work in all cases
    #current_crs = [*network.edges.crs.values()]
    #current_crs = str(current_crs[0])
    geometry = geometries[0]
    lat = pygeos.get_y(pygeos.centroid(geometry))
    lon = pygeos.get_x(pygeos.centroid(geometry))
    # formula below based on :https://gis.stackexchange.com/a/190209/80697 
    approximate_crs = "epsg:3035"# + str(int(32700-np.round((45+lat)/90,0)*100+np.round((183+lon)/6,0)))
    #from pygeos/issues/95
    coords = pygeos.get_coordinates(geometries)
    transformer=pyproj.Transformer.from_crs(current_crs, approximate_crs,always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    result = pygeos.set_coordinates(geometries, np.array(new_coords).T)
    return result

def return_all(x):
    return x

def intersect_country(country):
    
    data_path = r'/scistor/ivm/data_catalogue/open_street_map/'  #os.path.join('C:\\','Data')
    
    if not os.path.exists(os.path.join(data_path,'EU_flooded_road_networks','{}-edges.feather'.format(country))):
 
        flood_maps_path = os.path.join(data_path,'floodMaps_Europe_2018_mergedshifted')
        flood_maps = [os.path.join(flood_maps_path,x) for x in os.listdir(flood_maps_path)]

        aoi_maps_path = os.path.join(data_path,'floodMaps_Europe_2018_AreaofInfluence_shifted')
        aoi_maps = [os.path.join(aoi_maps_path,x) for x in os.listdir(aoi_maps_path)]
        
        road_network = os.path.join(data_path,'road_networks','{}-edges.feather'.format(country))
        
        road_df = pd.read_feather(road_network)
        road_df.geometry = pygeos.from_wkb(road_df.geometry)
        road_df.geometry = reproject(road_df.geometry.values)
        road_df = gpd.GeoDataFrame(road_df)
        
        for flood_map in flood_maps:
            returnp = flood_map.split('_')[5].split('.')[0]
            
            tqdm.pandas(desc=country+' '+returnp)
            
            flood_stats = road_df.geometry.progress_apply(lambda x: zonal_stats(x,flood_map,all_touched=True)) 
            road_df['{}_cells_intersect'.format(returnp)] = [x[0]['count'] for x in flood_stats]
            road_df['{}_min_flood_depth'.format(returnp)] = [x[0]['min'] for x in flood_stats]
            road_df['{}_max_flood_depth'.format(returnp)] = [x[0]['max'] for x in flood_stats]
            road_df['{}_mean_flood_depth'.format(returnp)] = [x[0]['mean'] for x in flood_stats]

        for aoi_map in aoi_maps:
            aoip = 'AoI_'+aoi_map.split('_')[6].split('.')[0]
            tqdm.pandas(desc=country+' '+aoip)
            
            aoi_stats = road_df.geometry.progress_apply(lambda x: zonal_stats(x,aoi_map,all_touched=True,stats="unique majority", add_stats={'all_uniq':return_all})) 
            road_df['{}_majority'.format(aoip)] = [x[0]['majority'] for x in aoi_stats]
            road_df['{}_unique'.format(aoip)] = [[y for y in np.unique(x[0]['all_uniq'].data) if y != 0] for x in aoi_stats]
            
        fds_map = os.path.join(data_path,'floodprotection','floodProtection_v2019_5km_nibbled.tif')
        fds_stats = road_df.geometry.progress_apply(lambda x: zonal_stats(x,fds_map,all_touched=True,stats="unique majority", add_stats={'all_uniq':return_all})) 
        road_df['fds_majority'] = [x[0]['majority'] for x in fds_stats]
        road_df['fds__unique'] = [[y for y in np.unique(x[0]['all_uniq'].data) if y != 0] for x in fds_stats]  


        road_df = pd.DataFrame(road_df)
        road_df.geometry = pygeos.to_wkb(pygeos.from_shapely(road_df.geometry))
        
        pd.DataFrame(road_df).to_feather(os.path.join(data_path,'EU_flooded_road_networks','{}-edges.feather'.format(country)))
        
if __name__ == "__main__":
    
    countries = ['ALB','AND','BGR','FIN','GIB','HRV','HUN','LIE','MLT','ROU','SRB','SVK','LUX','IRL',
             'BEL','NLD','DNK','FRA','ESP','PRT','SWE','NOR','CHE','AUT','ITA','CZH','POL','GRC','DEU','GBR']
    
    with Pool(5) as pool: 
        pool.map(intersect_country,countries,chunksize=1)   