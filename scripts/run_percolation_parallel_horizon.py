import pandas as pd
from pathos.multiprocessing import Pool
import os, sys
from random import shuffle
import pickle

sys.path.append(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\scripts")
from percolation_optimized_parallel import stochastic_network_analysis_phase1, stochastic_network_analysis_phase2, import_graph


class RunPercolation:
    def __init__(self, cntry_setup, countries, reps, output_folder):
        self.cntry_setup = cntry_setup
        self.countries = countries
        self.reps = reps
        self.output_folder = output_folder

    # prepare the pickles to do the parallel processing with
    def prep_par(self):
        print(len(self.countries), 'countries')
        for ctr in self.countries:
            combinations = [int(x) for x in self.cntry_setup.loc[self.cntry_setup['code3'] == ctr, 'aoi_combinations'].iloc[0].split(' ')]
            if ctr in ['BEL', 'DEU', 'NLD']:
                # For Belgium, Germany and the Netherlands we are using the NUTS-2 classification
                nuts_class = 'nuts2'
            else:
                # For the other countries we use the NUTS-3 classification
                nuts_class = 'nuts3'

            G = import_graph(ctr, nuts_class=nuts_class)
            for com in combinations:
                stochastic_network_analysis_phase1(G, com, self.reps, ctr, nuts_class)

    def run_par(self, nr_cores):
        # do the parallel processing
        cntries_todo = self.countries
        cntries_todo = [self.cntry_setup.loc[self.cntry_setup['code3'] == x, 'country'].iloc[0].lower() for x in cntries_todo]
        # shuffle(cntries_todo)
        for ctr in cntries_todo:
            # find all the folders, containing the scheduled combinations
            folders = os.listdir(
                os.path.join(self.output_folder.format(ctr), 'scheduled'))  # folder correspond to items of the combs_list defined before

            todo = []  # list of all tuples (combination, i(experiment ID), country name, country code (3 letters))
            for comb in folders:
                pkls = os.listdir(os.path.join(self.output_folder.format(ctr), 'scheduled', comb))  # list of pickles in each folder
                for pkl in pkls[:200]:  # Limit to 200 iterations for now
                    schedule_file = os.path.join(self.output_folder.format(ctr), 'scheduled', comb, pkl.split('.')[0] + '.pkl')
                    with open(schedule_file, 'rb') as f:
                        todo.append(pickle.load(f))

            shuffle(todo)
            # print(todo[0:10])  # just to check if we indeed shuffled
            print("In total we will do {} mini-processes".format(len(todo)))

            # Carry out the scheduled experiments
            with Pool(int(nr_cores)) as pool:
                pool.map(stochastic_network_analysis_phase2, todo, chunksize=1)


if __name__ == '__main__':
    cntrySetup = pd.read_csv(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\data\nuts3_combinations.csv")
    # countries_ = ['SVK', 'SRB', 'CZE', 'DNK', 'GRC', 'ESP', 'HUN', 'IRL', 'POL', 'AUT']  # 'BEL', 'DEU', 'NLD',
    countries_ = ['FRA']
    reps_ = 500
    outputFolder = r"P:\osm_flood\network_analysis\igraph\{}"

    running = RunPercolation(cntry_setup=cntrySetup, countries=countries_, reps=reps_, output_folder=outputFolder)
    if sys.argv[1] == 'prep_par':
        running.prep_par()
    elif sys.argv[1] == 'run_par':
        print("Using", sys.argv[2], "cores")
        running.run_par(sys.argv[2])
    else:
        print("wrong input, use 'prep_par' or 'run_par (nr of cores)'")
