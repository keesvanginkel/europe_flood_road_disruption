from pathos.multiprocessing import Pool
from random import shuffle
import json
from Europe_utils import *
from pathlib import Path

from percolation_event_based import *

if __name__ == '__main__':
    nr_cores = 6

    country_code = 'DEU'
    nuts_class = 'nuts2'


    event_folder = Path('D:\Europe_percolation\event_sampling_vs2\main_output\germany\scheduled')
    event_json_paths = list(event_folder.glob('year_*.json'))
    print('Todo: {} json dicts containing floods'.format(len(event_json_paths)))
    Sorted = []
    for filename in sorted(event_json_paths,
                           key=lambda path: int(path.stem.split('_')[1])):
        Sorted.append(filename)
    print(Sorted)
    config_file = 'config_eventbased_2.json'

    tuples = []
    #prepare all instruction for workers
    for json_path in Sorted:
        tuples.append((json_path,config_file,country_code,nuts_class))


    print('start evaluating events')

    with Pool(nr_cores) as pool:
        pool.map(call_evaluate_event, tuples, chunksize=1)

    print('Percolation analysis finished for:', ctr)

    #linear call of the function
    #evaluate_event(path_to_event_json=sample_file,config_file=config_file,country_code=country_code,nuts_class='nuts2')

