import pandas as pd
from pathos.multiprocessing import Pool
import os, sys
from random import shuffle
import pickle
import json

#Todo: fix this by giving package the appropriate name
#sys.path.append(r"P:\osm_flood\network_analysis\igraph\europe_flood_road_disruption\scripts")
from percolation_optimized_parallel import stochastic_network_analysis_phase1, stochastic_network_analysis_phase2
from percolation_optimized_parallel import import_graph, stochastic_network_analysis_phase1_event_sampling
from utils import load_config
from Europe_utils import *
from tqdm import tqdm


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
            if ctr in ['DEU', 'NLD','ITA','GBR']: #'BEL',
                # For Belgium, Germany and the Netherlands, Italy and UK we are using the NUTS-2 classification
                #... but for Belgium not in the uncertainty analysis :)
                nuts_class = 'nuts2'
            else:
                # For the other countries we use the NUTS-3 classification
                nuts_class = 'nuts3'

            G = import_graph(ctr, nuts_class=nuts_class,config_file=self.config)
            for com in combinations:
                #Todo: also give special settings to this function
                stochastic_network_analysis_phase1(self.config,G, com, self.reps, ctr, nuts_class)

        print('Preparing pickles finished for:',ctr)

    def prep_par_event_sampling(self,nuts_class = 'nuts3',sampling_json='entire_basin_sampling.json'):
        """
        Scheduler for preparing experiments from a timeseries of flood events,

        Used for Germany, guided sampling

        Arguments:
            *nuts_class* (string) : NUTS-level for running analysis, can be 'nuts3' or 'nuts2'
            *self.countries* (list of strings) : List of 3-letter codes of the countries to analyse e.g. ['BEL']
            *sampling_json* (string) : filename of the json file containing the aoi's per year
                                        (this .json file is made with visualisations/GFZ_JRC_datanew.ipynb)

        """

        ####### DEPRECIATED ######

        warnings.warn("""This version of the event sampling is depreciated, look into version 02 instead!!!""")

        assert nuts_class in ['nuts2','nuts3']
        if len(self.countries) > 1:
            warnings.warn("""This script is only suitable for running for one country at a time, 
                             will only run one country: {}""".format(self.countries[0]))

        #Open undisrupted network graph
        print('running.prep_par_event_sampling(): Start loading graph')
        #G = import_graph(self.countries[0], nuts_class=nuts_class, config_file=self.config)
        print('graph loaded')

        #Open json file that contains sampling procedure
        config = load_config(self.config)
        country_name = self.countries[0] # todo
        out_foldername =  country_name + nuts_class + '__sampling__'

        json_path = config['paths']['data'] / 'sampling_data' / sampling_json
        with open(json_path, 'r') as f:
            sampling_set = json.load(f)

        print('stop')
        for year in tqdm(sampling_set['data'].keys()):
            data = sampling_set['data'][year]
            #data['ds'] # *ds*     : discharge station at which floods occur
            #data['b_aoi'] # *b_aoi*  : basin aoi corresponding to discharge station
            #data['c_aoi'] # *c_aoi*  : all cell aois in this basin'
            stochastic_network_analysis_phase1_event_sampling(config_file,country_name, nuts_class,year,data)



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
                        inTuple = pickle.load(f) #load the original instruction from prep_par
                        outTuple = tuple([self.config] + list(inTuple) + [self.special_setting])
                        todo.append(outTuple)

            shuffle(todo)
            print(todo[0:10])  # just to check if we indeed shuffled
            print("In total we will do {} mini-processes".format(len(todo)))

            # Carry out the scheduled experiments
            print('Run_par() starting scheduled experiments for {}'.format(ctr))
            #with Pool(int(nr_cores)) as pool:
                #pool.map(stochastic_network_analysis_phase2, todo, chunksize=1)
            stochastic_network_analysis_phase2(todo[0]) #useful for bugfixing
            print('Percolation analysis finished for:', ctr)





if __name__ == '__main__':
    #RUN THIS FOR REGULAR ANALYSIS AND UNCERTAINTY ANALYSIS
    #countries_ = N0_to_3L(['LT','LV','DK','MK','SI']) #Provide list of 3l-codes
    countries_ = [N0_to_3L('HU')]
    nuts_level = 'nuts3'
    reps_ = 200 #Repetitions per #AoIs
    constrain_reps_ = 10 #Schedule all, but only run these first.

    #Read the set-up per country
    config_file = 'config.json'
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
                             output_folder=outputFolder,config=config_file,special_setting=None)

    #running.prep_par()
    running.run_par(nr_cores=7)

    # if sys.argv[1] == 'prep_par':
    #     running.prep_par()
    # elif sys.argv[1] == 'run_par':
    #     print("Using", sys.argv[2], "cores")
    #     running.run_par(sys.argv[2])
    # else:
    #     print("wrong input, use 'prep_par' or 'run_par (nr of cores)'")

if False:
#if __name__ == '__main__':
    from percolation_event_based import *

    ### RUN THIS FOR EVENT SAMPLING ANALYSIS (version 2)
    countries_ = ['DEU']
    nuts_level = 'nuts2'
    config_file = 'config_eventbased_2.json'
    #special_setting = 'event_based' #still unused

    #Some unused declarations for initializing RUNNING object
    reps_ = None
    outputFolder = str(config['paths']['main_output']) + r'\{}'

    #Todo: do we need the below code for event-based sampling???
    cntrySetup_path = config['paths']['data'] / '{}_combinations.csv'.format(nuts_level)
    if not cntrySetup_path.exists():
        raise OSError(2, 'Cannot find the file prescribing the AoI sampling:', '{}'.format(cntrySetup_path))
    else:
        print('Reading the AoI sampling procedure from: {}'.format(cntrySetup_path))
    cntrySetup = pd.read_csv(cntrySetup_path, sep=';')

    config = load_config(file=config_file)
    # Run a small test to check if all the paths are well configured:
    for key, path in config['paths'].items():
        print(key, path, path.exists())

    #Start with actual preparations
    #note that some of these are not used


    #Prepare experiments
    #running.prep_par_event_sampling(nuts_class=nuts_level, sampling_json='entire_basin_sampling.json')

    #Run experiments


