import pandas as pd
from pathos.multiprocessing import Pool
import os, sys
from random import shuffle
import pickle

#Todo: fix this by giving package the appropriate name
#sys.path.append(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\scripts")
from percolation_optimized_parallel import stochastic_network_analysis_phase1, stochastic_network_analysis_phase2, import_graph
from utils import load_config
from Europe_utils import *


class RunPercolation:
    def __init__(self, cntry_setup, countries, reps, output_folder):
        self.cntry_setup = cntry_setup
        self.countries = countries
        self.reps = reps
        self.output_folder = output_folder

    # prepare the pickles to do the parallel processing with
    def prep_par(self):
        print('prep_par(), preparing pickles for parallel processing for ',len(self.countries), 'countries')
        for ctr in self.countries:
            combinations = [int(x) for x in self.cntry_setup.loc[self.cntry_setup['code3'] == ctr, 'aoi_combinations'].iloc[0].split(' ')]
            if ctr in ['DEU', 'NLD','ITA','GBR']: #'BEL',
                # For Belgium, Germany and the Netherlands, Italy and UK we are using the NUTS-2 classification
                #... but for Belgium not in the uncertainty analysis :)
                nuts_class = 'nuts2'
            else:
                # For the other countries we use the NUTS-3 classification
                nuts_class = 'nuts3'

            G = import_graph(ctr, nuts_class=nuts_class)
            for com in combinations:
                stochastic_network_analysis_phase1(G, com, self.reps, ctr, nuts_class)

        print('Preparing pickles finished for:',ctr)


    def run_par(self, nr_cores):
        # do the parallel processing
        cntries_todo = self.countries
        cntries_todo = [self.cntry_setup.loc[self.cntry_setup['code3'] == x, 'country'].iloc[0].lower() for x in cntries_todo]

        # shuffle(cntries_todo)
        for ctr in cntries_todo:
            print('Starting run_par() for countries {}'.format(ctr))

            # find all the folders, containing the scheduled combinations
            folders = os.listdir(
                os.path.join(self.output_folder.format(ctr), 'scheduled'))  # folder correspond to items of the combs_list defined before

            todo = []  # list of all tuples (combination, i(experiment ID), country name, country code (3 letters))
            for comb in folders:
                pkls = os.listdir(os.path.join(self.output_folder.format(ctr), 'scheduled', comb))  # list of pickles in each folder

                for pkl in pkls: #pkls[:constrain_reps_]  # If lower than scheduled, limit to pkls[:200] 200 iterations for now

                    schedule_file = os.path.join(self.output_folder.format(ctr), 'scheduled', comb, pkl.split('.')[0] + '.pkl')
                    with open(schedule_file, 'rb') as f:
                        todo.append(pickle.load(f))

            #shuffle(todo)
            print(todo[0:10])  # just to check if we indeed shuffled
            print("In total we will do {} mini-processes".format(len(todo)))

            # Carry out the scheduled experiments
            print('Run_par() starting scheduled experiments for {}'.format(ctr))
            with Pool(int(nr_cores)) as pool:
                pool.map(stochastic_network_analysis_phase2, todo, chunksize=1)
            #stochastic_network_analysis_phase2(todo[0]) #useful for bugfixing
            print('Percolation analysis finished for:', ctr)


if __name__ == '__main__':
    #countries_ = N0_to_3L(['LT','LV','DK','MK','SI']) #Provide list of 3l-codes
    #countries_ = [N0_to_3L('BE')]
    countries_ = ['BEL']
    nuts_level = 'nuts3'
    reps_ = 500 #Repetitions per #AoIs
    #constrain_reps_ = 200 #Schedule all, but only run these first.

    #Read the set-up per country
    config = load_config()
    #Run a small test to check if all the paths are well configured:
    for key, path in config['paths'].items():
        print(key, path, path.exists())

    cntrySetup_path = config['paths']['data'] / '{}_combinations.csv'.format(nuts_level)
    if not cntrySetup_path.exists():
        raise OSError(2, 'Cannot find the file prescribing the AoI sampling:', '{}'.format(cntrySetup_path))
    else: print('Reading the AoI sampling procedure from: {}'.format(cntrySetup_path))
    cntrySetup = pd.read_csv(cntrySetup_path,sep=';')

    outputFolder = str(config['paths']['main_output']) + r'\{}'
    print(outputFolder)

    running = RunPercolation(cntry_setup=cntrySetup, countries=countries_, reps=reps_, output_folder=outputFolder)

    #running.prep_par()
    running.run_par(8)

    # if sys.argv[1] == 'prep_par':
    #     running.prep_par()
    # elif sys.argv[1] == 'run_par':
    #     print("Using", sys.argv[2], "cores")
    #     running.run_par(sys.argv[2])
    # else:
    #     print("wrong input, use 'prep_par' or 'run_par (nr of cores)'")

