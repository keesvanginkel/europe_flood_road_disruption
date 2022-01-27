"""This script is used for aggregating in the postprocessing step. """

import visualisations_new as vis
from utils import load_config

config = load_config(file='config.json')


#vis.aggregate_results_step1(select=['hungary','netherlands'],config=config) #to .csv per # combinations
vis.aggregate_results_step2(select=['hungary','netherlands'],config=config)

