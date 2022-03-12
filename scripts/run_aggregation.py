"""This script is used for aggregating in the postprocessing step. """

import visualisations_new as vis
from utils import load_config

config = load_config(file='config_sens.json')

select = ['belgium_waterdepth_50cm']
#vis.aggregate_results_step1(select=select,config=config) #to .csv per # combinations
vis.aggregate_results_step2(select='all',config=config)

