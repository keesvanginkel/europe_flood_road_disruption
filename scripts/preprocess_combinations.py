# -*- coding: utf-8 -*-
"""
Created on 21-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import os
import geopandas as gpd
from math import comb
import pandas as pd

# translation between countrycodes (2- and 3-letter and country names)
translate_cntr_codes = pd.read_csv(r"D:\COACCH_paper\europe_flood_road_disruption\data\country_codes.csv",
                                delimiter=';').set_index('code2').to_dict(orient='dict')

# set paths
country = 'NL'
input_folder = r"D:\COACCH_paper\data"
current_country = translate_cntr_codes['country'][country].lower()  # The country that is analysed
print("\nCurrent iteration is for:", current_country)
output_folder = r"D:\COACCH_paper\data\output\{}".format(current_country)

# NUTS-3 regions of Europe
nuts_3_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_3.shp"
nuts_2_regions = r"D:\COACCH_countries\countries_shp\NUTS_RG_01M_2016_3035_LEVL_2.shp"

nuts = gpd.read_file(nuts_2_regions)
count_nuts = nuts.groupby('CNTR_CODE').size().to_frame('nr_nuts').reset_index()
count_nuts.rename({'CNTR_CODE': 'code2'}, axis=1, inplace=True)
count_nuts['nr_routes'] = count_nuts['nr_nuts'].apply(lambda x: comb(x, 2))
count_nuts['country'] = count_nuts['code2'].apply(lambda x: translate_cntr_codes['country'][x])
count_nuts['code3'] = count_nuts['code2'].apply(lambda x: translate_cntr_codes['code3'][x])
count_nuts.to_csv(r"D:\COACCH_paper\europe_flood_road_disruption\data\nuts2_combinations.csv")

networks_europe_path = os.path.join(input_folder, 'networks_intersect_hazard_elco_koks')
edge_file = [os.path.join(networks_europe_path, f) for f in os.listdir(networks_europe_path) if
            f == country + '-edges.feather'][0]

# read the network files from Elco Koks
network = pd.read_feather(edge_file)

all_aois = list(network['AoI_RP100y_unique'])
all_aois = [int(x) for l in all_aois for x in l if (x != 0) and (x == x)]
all_aois = list(set(all_aois))

list_combinations = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
max_aoi = len(all_aois)
list_combinations = [x for x in list_combinations if x < max_aoi]
list_combinations.append(max_aoi)
list_combinations = [str(x) for x in list_combinations]

df = pd.read_csv(r"D:\COACCH_paper\europe_flood_road_disruption\data\nuts3_combinations.csv")
df.loc[df['code2'] == country, 'aoi_combinations'] = " ".join(list_combinations)
df.to_csv(r"D:\COACCH_paper\europe_flood_road_disruption\data\nuts3_combinations.csv")
