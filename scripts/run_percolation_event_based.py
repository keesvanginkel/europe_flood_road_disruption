from pathos.multiprocessing import Pool
from random import shuffle
import json
from Europe_utils import *
from pathlib import Path
from utils import load_config

from percolation_event_based import *

if __name__ == '__main__':
    nr_cores = 6

    country_code = 'DEU'
    ctr = country_names(country_code)
    nuts_class = 'nuts2'


    event_folder = Path('D:\Europe_percolation\event_sampling_vs3\main_output\germany\scheduled')
    event_json_paths = list(event_folder.glob('year_*.json'))
    print('Todo: {} json dicts containing floods'.format(len(event_json_paths)))
    Sorted = []
    for filename in sorted(event_json_paths,
                           key=lambda path: int(path.stem.split('_')[1])):
        Sorted.append(filename)
    #print(Sorted)
    config_file = 'config_eventbased_3.json'
    config = load_config(file=config_file)

    tuples = []


    ## optimisation (version > 1.0): load graph at this stage, and give it to workers
    graph_pickle = config['paths']['preproc_output'] / ctr / (ctr + '_G.pkl')
    G = ig.Graph.Read_Pickle(graph_pickle)

    #from percolation_event_based import import_graph_v3 #Dit werkt niet want verkeerde metadata
    #G = import_graph_v3(country_code,nuts_class,config_file)

    print('start evaluating events')

    # prepare all instruction for workers
    for json_path in Sorted:
        tuples.append((json_path, config_file, country_code, nuts_class))

    # with Pool(nr_cores) as pool:
    #     pool.map(call_evaluate_event, tuples, chunksize=1)


    #linear call of the function
    shuffle(Sorted)
    for event in Sorted:
        evaluate_event_3(path_to_event_json=event,config_file=config_file,
                         country_code=country_code,nuts_class='nuts2',G=G.copy())

    print('Percolation analysis finished for:', ctr)