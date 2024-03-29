import igraph as ig
import json
import logging
import os, sys
import pandas as pd
from pathos.multiprocessing import Pool
from random import shuffle
import pickle
from pathlib import Path
import random
from tqdm import tqdm
import time

from percolation_optimized_parallel import stochastic_network_analysis_phase1, stochastic_network_analysis_phase2
from percolation_optimized_parallel import import_graph, stochastic_network_analysis_phase1_event_sampling
from utils import load_config
from Europe_utils import *


class RunPercolation:
    def __init__(self, cntry_setup, countries, reps, output_folder,config='config.json',special_setting=None):
        self.cntry_setup = cntry_setup
        self.countries = countries
        self.reps = reps
        self.output_folder = output_folder
        self.config = config #todo: I don't think this information needs to be passed to thte methods explicitly
        self.special_setting = special_setting

    # prepare the pickles to do the parallel processing with
    # Todo: now that we are using a config file, this script can be made simpler
    def prep_par(self):
        """
        Scheduler for preparing a regular percolation analysis
        actual work is done by stochastic_network_analysis_pahse1 (in percolation_optimized_parallel)

        Arguments:
            *self.countries* (list of strings) : List of 3-letter codes of the countries to analyse e.g. ['BEL']
            *self.config* (string) : path to configuration file in the main folder e.g. 'config.json'
            *self.cntry_setup* (pd DataFrame) : DataFrame containing the percolation setup
                                                    (loaded from data/NUTSX_combinations.csv)
            *self.reps* (int) : number of repetitions per combinations of microfloods
            *self.output_folder* : output folder for the analysis
            *self.special_setting* : run if you want to run in a special setting

        Returns:
            None

        Effect:
            Creates a folder with the country name in 'main_output', with a subfolder 'scheduled',
            with subsubfolders with the nr of AoIs to sample at the same time (i.e. the nr_comb)
            and puts pickles in these folders
            These pickles contain several variables needed to carry out that experiment

        """
        print('prep_par(), preparing pickles for parallel processing for ',len(self.countries), 'countries')
        for ctr in self.countries:
            combinations = [int(x) for x in self.cntry_setup.loc[self.cntry_setup['code3'] == ctr, 'aoi_combinations'].iloc[0].split(' ')]
            combinations = [c for c in combinations if c != 0]
            #todo: make sure that in preprocessing returning c = 0 no longer happens!
            if ctr in ['DEU', 'NLD','ITA','GBR']: #'BEL',
                # For Belgium, Germany and the Netherlands, Italy and UK we are using the NUTS-2 classification
                #... but for Belgium not in the uncertainty analysis :)
                nuts_class = 'nuts2'
            else:
                # For the other countries we use the NUTS-3 classification
                nuts_class = 'nuts3'

            #todo: this can be made even faster by directly loading the pickled graph (from version 1.1 onwards)
            #G = import_graph(ctr, nuts_class=nuts_class,config_file=self.config)
            country_name = country_names(ctr).lower()
            graph_pickle = config['paths']['preproc_output'] / country_name / (country_name + '_G.pkl')
            G = ig.Graph.Read_Pickle(graph_pickle)
            for com in combinations:
                #Todo: also give special settings to this function
                stochastic_network_analysis_phase1(self.config,G, com, self.reps, ctr, nuts_class)

        print('Preparing pickles finished for:',ctr)

    def run_par(self, nr_cores,run_mode_,route_algorithm_):
        """"
        Todo: improve docstring

        """

        # do the parallel processing
        cntries_todo = self.countries
        cntries_todo = [self.cntry_setup.loc[self.cntry_setup['code3'] == x, 'country'].iloc[0].lower() for x in cntries_todo]

        # shuffle(cntries_todo)
        for ctr in cntries_todo:
            print('Starting run_par() for country {}'.format(ctr))


            # read folder structure created with prep_par()
            outputFolder = Path(self.output_folder.format(ctr)) / 'scheduled'
            if not outputFolder.exists():
                raise FileNotFoundError(
                    'outputFolder {} does not exist, did you forget to run prep_par?'.format(outputFolder))

            # find all the folders, containing the scheduled combinations
            folders = [x for x in outputFolder.iterdir() if x.is_dir()]  # folders correspond to items of the combs_list

            # read one example tuple to figure out the desired nutsclass
            example_pickle = [p for p in folders[0].iterdir() if p.suffix == '.pkl'][0]
            with open(example_pickle, 'rb') as f:
                example_pickle = pickle.load(f)
            nutsClass = example_pickle[5]
            assert nutsClass in ['nuts2','nuts3'] #check if a valid value is drawn from the pickle
            #todo: check that this corresponds with the other settings given

            # INSTANTIATE LOGGING FUNCTIONALITY
            logger_filename = config['paths']['logs'] / 'log_{}_{}_{}_{}.log'.format(
                ctr,nutsClass,Path(__file__).stem,'run_par')
            rootLogger = make_rootLogger(logger_filename)

            # optimisation (version > 1.0): load graph at this stage, and give it to workers
            country_code3 = country_code_from_name(ctr,l3=True)

            graph_pickle = config['paths']['preproc_output'] / ctr / (ctr + '_G.pkl')
            G = ig.Graph.Read_Pickle(graph_pickle)
            #Depreciated from version > 1.0
            #G = import_graph(country_code3, nuts_class=nutsClass, config_file=config_file)
            check_n_es = len(G.es)
            check_n_vs = len(G.vs)
            rootLogger.info('Reference nr edges|vertices original graph: {} | {}'.format(check_n_es,check_n_vs))

            if run_mode_[0] == 'single' and run_mode_[1] == 'random':
                #"""Run a random experiment for each selected country"""
                rootLogger.info('Running a random experiment for {}'.format(ctr))
                comb = random.choice(folders)
                pkls = [pkl for pkl in comb.iterdir()]
                pkl = random.choice(pkls)
                with open(pkl, 'rb') as f:
                    inTuple = pickle.load(f)  # load the original instruction from prep_par
                outTuple = tuple(
                    [self.config] + list(inTuple) + [self.special_setting] + [G] + [check_n_es] + [route_algorithm_])

                t10 = time.time()
                stochastic_network_analysis_phase2(outTuple)  # useful for bugfixing
                t11 = time.time()
                rootLogger.info('one experiment for {} costed {} s'.format(ctr, t11 - t10))
                return None

            if run_mode_[0] == 'single' and run_mode_[1] != 'random':
                if not len(run_mode_) == 3:
                    raise ValueError("Expection tuple length 3 ('single',nr_comb,i)")
                rootLogger.info('Running one selected experiment {}'.format(ctr))
                search_nr_comb = run_mode_[1]
                search_i = run_mode_[2]
                rootLogger.info('Start searching for experiment {} {}'.format(search_nr_comb,search_i))
                folder = [f for f in folders if int(f.stem) == search_nr_comb][0]
                pkl = [pkl for pkl in folder.iterdir() if int(pkl.stem) == search_i][0]
                with open(pkl, 'rb') as f:
                    inTuple = pickle.load(f)
                outTuple = tuple(
                        [self.config] + list(inTuple) + [self.special_setting] + [G] + [check_n_es] + [route_algorithm_])
                t10 = time.time()
                stochastic_network_analysis_phase2(outTuple)  # useful for bugfixing
                t11 = time.time()
                rootLogger.info('one experiment for {} costed {} s'.format(ctr, t11 - t10))

            #todo: reps constraint
            if run_mode_[0] == 'linear':
                rootLogger.info('Start looping over {}'.format(ctr))
                shuffle(folders)
                for comb in folders:
                    pkls = [pkl for pkl in comb.iterdir() if pkl.suffix == '.pkl']
                    shuffle(pkls)
                    for pkl in pkls:
                        with open(pkl, 'rb') as f:
                            inTuple = pickle.load(f)
                        outTuple = tuple(
                            [self.config] + list(inTuple) + [self.special_setting] + [G.copy()] + [check_n_es] + [route_algorithm_])
                        stochastic_network_analysis_phase2(outTuple)
                rootLogger.info('Percolation analysis finished for: {}'.format(ctr))

            if run_mode_[0] == 'parallel':
                rootLogger.info('Start parallel processing of {}, by preparing scheduled experiments'.format(ctr))
                todo = [] # list of all tuples (combination, i(experiment ID), country name, country code (3 letters))
                for comb in folders:
                    pkls = [pkl for pkl in comb.iterdir() if pkl.suffix == '.pkl']
                    for pkl in pkls:
                        with open(pkl, 'rb') as f:
                            inTuple = pickle.load(f)
                        outTuple = tuple(
                            [self.config] + list(inTuple) + [self.special_setting] + [G] + [check_n_es] + [route_algorithm_])
                        todo.append(outTuple)
                rootLogger.info('In total we will do {} mini-processes.'.format(len(todo)))

                #Carry out the scheduled experiments
                rootLogger.info('Starting pools for {}, with {} cores'.format(ctr,nr_cores))
                shuffle(todo)
                with Pool(int(nr_cores)) as pool:
                    pool.map(stochastic_network_analysis_phase2, todo, chunksize=1)

                rootLogger.info('Percolation analysis finished for: {}'.format(ctr))




def make_rootLogger(filename):
    """
    Returns root logger object, for a specific country
        *filename* (Path) : the path to the desired logfile

    Effect: log messages are printed to terminal; log is saved in config -> paths -> logs dir

    """
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(funcName)20s()] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(__name__)
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.info('Logger is created')
    return rootLogger

if __name__ == '__main__':
    #RUN THIS FOR REGULAR ANALYSIS AND UNCERTAINTY ANALYSIS
    #countries_ = N0_to_3L(['LT','LV','DK','MK','SI']) #Provide list of 3l-codes
    #countries_ = ['ALB', 'AUT', 'BGR', 'CHE', 'CZE', 'DNK', 'EST','ESP', 'FIN']

    #countries_ = ['FIN', 'FRA', 'GRC',
    #'HRV', 'HUN', 'IRL', 'LTU', 'LVA', 'MKD',  'POL', 'PRT', 'ROU', 'SRB', 'SVK', 'SVN', 'SWE'] #'NOR'
    #countries_ = ['BEL','NLD','GBR','ITA','DEU'] #NUTS-2 countries



    countries_ = ['BEL']
    #shuffle(countries_)


    nuts_level = 'nuts3'
    reps_ = 200 #Repetitions per #AoIs
    #Todo: cleanup constrain_reps_
    constrain_reps_ = 20 #Schedule all, but only run these first.
    #run_mode_ = ('single','random')
    #run_mode_ = ('single',6,130)
    run_mode_ = ('linear',)
    #run_mode_ = ('parallel',)

    #Select algorithm to run model:
    route_algorithm_ = ('version_2')

    #Read the set-up per country
    config_file = 'config_sens.json'
    config = load_config(file=config_file)
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

    running = RunPercolation(cntry_setup=cntrySetup, countries=countries_, reps=reps_,
                             output_folder=outputFolder,config=config_file,special_setting='depth_threshold')

    #running.prep_par()
    running.run_par(nr_cores=4,run_mode_=run_mode_,route_algorithm_=route_algorithm_)

