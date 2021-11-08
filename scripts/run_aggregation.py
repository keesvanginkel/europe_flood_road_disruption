"""This script is used for aggregating in the postprocessing step. """

import visualisations_new as vis
from utils import load_config

config = load_config(file='config_agg_unc.json')

#ignore = ['austria_buffer', 'belgium_0cm', 'belgium_50cm', 'belgium_distance', 'belgium_fla',  #'belgium_giant_component',
#          'belgium_n3_alllevels', 'belgium_n3_pri', 'belgium_n3_prisec', 'belgium_shifted_centroids', 'belgium_wal',
#          'benelux', 'latvia_giant_component', 'poland_2000', 'rhine_alpine_corridor']

#ignore = ['albania', 'austria', 'belgium', 'bulgaria', 'croatia', 'czechia', 'denmark', 'estonia', 'figures',
#           'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy',  'latvia_old', #latvia
#           'lithuania', 'macedonia', 'netherlands', 'norway', 'poland', 'portugal', 'romania', 'serbia',
#           'slovakia', 'slovenia', 'spain', 'sweden', 'switzerland', 'united kingdom']

ignore= ['austria_buffer', 'belgium_0cm', 'belgium_50cm',  'belgium_distance', 'belgium_fla', 'belgium_buffer',
         'belgium_giant_component_part1', 'belgium_giant_component_part2', #'belgium_giant_component_all',
         'belgium_n3_alllevels', 'belgium_n3_pri', 'belgium_n3_prisec', 'belgium_shifted_centroids',
         'belgium_wal', 'benelux', 'germany', 'latvia_giant_component', 'poland_2000', 'rhine_alpine_corridor']

#vis.aggregate_results_step1(ignore=ignore,config=config) #to .csv per # combinations
vis.aggregate_results_step2(ignore=ignore,config=config)

