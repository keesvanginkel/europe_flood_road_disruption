"""
Created on Jan 26

This file is meant to eventually replace the visualisations.py

@Author Frederique de Groen and Kees van Ginkel

"""
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from warnings import warn

import Europe_utils as eu
from utils import load_config
import pdb as pdb
config_file = 'config.json' #also declared at the top of the file (ugly fix)
config = load_config(config_file)

#Suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# n,th Percentile
def q_n(x,n):
    """Returns the nth percentile of a series """
    return x.quantile(n)

def q_05(x):
    return(q_n(x,0.05))

def q_25(x):
    return(q_n(x,0.25))

def q_75(x):
    return(q_n(x,0.75))

def q_95(x):
    return(q_n(x,0.95))

def group_operations():
    "This returns a list with the summary statistics operations that are used for the aggregated lineplot vis."
    return ['min', q_05, q_25, 'mean', q_75, q_95, 'max']

group_operations = group_operations()



# Everything below  is a sort preprocessing step, before the actual plotting starts
def aggregate_results_step1(select = 'all', ignore = [None],config=None):
    """
    Function to aggregate the raw results of the percolation analysis
    
    Arguments:
        *select* ('all') or (list of countries) : if not 'all': only analyse these countries
        *ignore* (list of countries) : list of countries that you don't want to analyse e.g. ['denmark','latvia']
        *config* (configuration file) : output of load_config; if not provided, will load default config.json
    Effect:
        Aggregates the .csv files to higher levels in three steps
    """
    if not config:
        config = load_config()
    country_results_folder = config['paths']['main_output']
    #Example folder structure: country_results_folder/albania/finished/

    #Step 1: for each country, summarize individual samples to results per combination
    print('aggregate_results(): Starting step 1/2:')
    countries_paths = [x for x in country_results_folder.iterdir() if x.is_dir()]
    countries = [x.stem for x in countries_paths]
    print('Identified {} countries'.format(len(countries)))
    if select != 'all':
        print('To aggregate: {}'.format(select))
    else:
        print('To aggregate: {}'.format(countries))
    print('To ignore: {}'.format(ignore))
    while True:
        if input('Do You Want To Continue? Press y to continue.') != 'y':
            break
        for path in countries_paths:
            if path.stem in ignore:
                print('ignoring {}'.format(path))
            else:
                if select != 'all':
                    if path.stem not in select:
                        print('{} not selected'.format(path.stem))
                        continue
                print('Start aggregating {}'.format(path))
                finished_folder = path / 'finished'
                if not finished_folder.exists():
                    raise OSError(2, 'Cannot find the folder containing \
                                  the raw results of the percolation analysis:',
                                  '{}'.format(path))
                combine_finished_stochastic(finished_folder)
                print('{}: Aggregated raw pickles per experiment, to .csv per # combinations'.format(path.stem))

def aggregate_results_step2(select = 'all',ignore = [None],config=None):
    """
    Function to aggregate the raw results of the percolation analysis

    Arguments:
        *select* ('all') or (list of countries) : if not 'all': only analyse these countries
        *ignore* (list of countries) : list of countries that you don't want to analyse e.g. ['denmark','latvia']
        *config* (configuration file) : output of load_config; if not provided, will load default config.json
    Effect:
        Aggregates the .csv files to higher levels in three steps
    
    """
    
    #Step 2a: for each country, summarize combinations in a dataframe
    if not config:
        config = load_config(config_file)
    country_results_folder = config['paths']['main_output']
    
    print('Starting step 2:')
    countries_paths = [x for x in country_results_folder.iterdir() if x.is_dir()]
    countries = [x.stem for x in countries_paths if x.stem not in ignore]
    if select != 'all':
        countries = [c for c in countries if c in select]
        folders = [df_stochastic_results(folder=country_results_folder / c / 'finished') for c in countries]
    dict_dfs = dict(zip(countries, folders)) #keys are countries, values dataframes with results

    #Step 2b: Summarize the results of all countries
    #folder_results = r'D:\COACCH_paper\data\output\{}'
    print('Starting step 2b:')
    # group the dataframes
    for c in countries:
        temp_df = dict_dfs[c]
        temp_df['country'] = c.capitalize()
        dict_dfs[c] = temp_df

    df = pd.concat(list(dict_dfs.values()), sort=False)
    df.to_csv(country_results_folder / 'all_combinations.csv',sep=';')

def combine_finished_stochastic(finished_folder):
    """Combines the separate csv files create by the parallel processed stochastic results.

    Arguments:
        input_path (string): path to the folder where the separate csv's are saved. e.g. 'albania/finished'

    Returns:
        None

    Effect:
        Will create summary of all samples per #combinations in the finished_folder
    """
    #assert input is a path
    if any([p.suffix == '.csv' for p in finished_folder.iterdir()]):
        raise Exception('This folder already contains .csv files, did you already run this script?')

    for folder in finished_folder.iterdir():
        if folder.is_dir():
            files = [f for f in folder.iterdir()]
            df = pd.read_csv(files[0],sep=';')  #avoid empty column
            #df.index = [int(files[0].stem)] #use filename as index
            for file in files[1:]:
                df_add = pd.read_csv(file,sep=';')
                #df_add.index = [int(file.stem)]  # use filename as index
                df = pd.concat([df, df_add], sort='False')

            columns = list(df.columns)

            column_order = ['AoI combinations','experiment','disrupted','no detour','avg extra time','AoI removed',
                            'OD-disrupted', 'OD-with_detour', 'with_detour_extra_times']
            if 'edges_in_giant' in columns:
                column_order.extend(['edges_in_graph','edges_in_giant'])

            df = df.reindex(columns=column_order)
            df = df.sort_values(by='experiment')
            df.to_csv(finished_folder / "aoi_{}.csv".format(folder.stem),sep=';',index=False)


        print('Combine_finished_stochastic finished for {}'.format(folder))

def df_stochastic_results(folder):
    """
    Summarizes the different combinations in a dataframe for the country

    Arguments:
        *country_folder* (Path) : finished results of a country, e.g.
                    country_results_folder/albania/finished

    Returns:
        *df* (DataFrame) : contains the results per #combinations (rows)

    """
    files = [f for f in folder.iterdir() if (f.suffix == '.csv')]

    # load data
    df = pd.read_csv(files[0],sep=';')
    for f in files[1:]:
        df_new = pd.read_csv(os.path.join(folder, f),sep=';')
        df = pd.concat([df, df_new], sort=False) #ignore_index=True

    df['AoI combinations'] = df['AoI combinations'].astype(int)
    df = df.sort_values(by=['AoI combinations','experiment'])
    return df

######################################## SOME MAIN FUNCTIONALITY NEEDED FOR MANY VISUALISATIONS ##########################
def main(config,constrain_reps=None):
    """
    Reads the (merged) outputs of the percolation analysis, and does some basic assembly work which is useful
    for many visualisations.

    Arguments:
        *config* (dict) : containing the configuration paths
        *constrain_reps* (int) : constrain the number of combinations per numer of aois (default=None)

    Returns:
        *df* (DataFrame) : the raw results
        *df_abs* (DataFrame) : results grouped by combi of absolute AOI AND country
        *df_rel* (DataFrame) : results grouped by combi of relative AOI and country
    """
    print(' -------- main() starting --------')
    folder_results = config['paths']['main_output']

    # READ SOURCE FILE
    csv_file = folder_results / 'all_combinations.csv'
    if not csv_file.exists():
        raise OSError(2, """Cannot find the file with the aggregated results, 
                            maybe you need to run aggregate_results_step2() first!
                            Missing: """, csv_file)

    df = pd.read_csv((csv_file),index_col=0,sep=';')
    #df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'])
    print('Succesfully loaded source file as dataframe, with columns:')
    print(df.columns)
    print('Available for {} countries'.format(len(df.country.unique())))

    available_countries = df.country.unique()

    print('Grouping per AoI-country combination')
    max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()

    #If requested (kwargs), only select the first n repetitions per country
    if constrain_reps is not None:
        warn('Number of repetitions per #aoi is constrained to {}'.format(constrain_reps))

        #create a copy of df
        df2 = df.copy()
        df = pd.DataFrame(columns=df2.columns)

        for cntr in available_countries:
            df_temp = select_n_reps(df2.loc[df2['country'] == cntr.capitalize()], constrain_reps)

            #Determine if there are enough samples
            comb_per_aoi = df_temp.groupby('AoI combinations').size()
            missing = comb_per_aoi.loc[comb_per_aoi != constrain_reps] #pd Series; index = nr AoI combs; value = reps
            for nr_aoi, reps in missing.items():
                max_aoi = max_aoi_comb[cntr.capitalize()]
                if nr_aoi == max_aoi:
                    pass #do nothing, because it is expected that only 1 combination exists for all AoIs at the same time
                elif nr_aoi == 1:
                    pass #do nothing, because this is expected if constrain_reps > total aoi in country
                else:
                    warn('{}: Only {} repetitions for {} AoIs at the same time'.format(cntr,reps,nr_aoi))

            df = df.append(df_temp)

    #Calculate all relative AoI scores.
    for cntr in available_countries :
        df.loc[df['country'] == cntr.capitalize(), 'AoI relative combinations'] = \
            df.loc[df['country'] == cntr.capitalize(), "AoI combinations"] / \
            max_aoi_comb[cntr.capitalize()] * 100

    # Groups unique combinations of #AoI combination and country (stats are about %OD pairs disrupted)
    df_abs = df.groupby(['AoI combinations', 'country'])['disrupted'].agg(group_operations).reset_index()
    df_rel = df.groupby(['AoI relative combinations', 'country'])['disrupted'].agg(group_operations).reset_index()

    print(' -------- main() finished --------')
    return(df,df_abs,df_rel)

def select_n_reps(df_country, n):
    """
    Select the first n repetitions from the percolation analysis and returns the input dataframe with only this selection

    e.g. df_country = df.loc[df['country'] == 'Belgium'] #with df being the raw percolation result from main()
         n = 200

    returns: *sel* (DataFrame) - similar to df_country, but only the first n reps per aoi

    """
    sel = pd.DataFrame(columns=df_country.columns)
    aois_unique = sorted(list(df_country['AoI combinations'].unique()))
    for i, aois in enumerate(aois_unique):
        # print(i,aois)
        subsel = df_country[df_country['AoI combinations'] == aois].reset_index(drop=True)
        # select top-n valuesd
        if subsel.shape[0] >= n:
            subsel = subsel.iloc[0:n]
        # print(subsel.shape)
        sel = sel.append(subsel)
    sel = sel.reset_index(drop=True)
    return sel

def interp_rel_aoi(df_rel,col,country,percentages):
    """
    Linearly interpolate the dataframe with relative AoI combinations to find given percentages
    
    Arguments:
        *df_rel* (DataFrame) : output of main()
        *col* (str) : name of the column to interp. 
                e.g. 'min', 'q_05', 'q_25', 'mean', 'q_75', 'q_95', 'max'
        *country* (str) : full country (e.g. 'Belgium')
        *percentages* (list of x-coordinates, i.e. the percentages for which to find the metric score

    Returns:
        *interpolated* (np Array) : the interpolated values
    """

    df_rel_c = df_rel.loc[df_rel['country'] == country]
    df_raw = df_rel_c[['AoI relative combinations',col]].set_index('AoI relative combinations')
    arr = df_rel_c[['AoI relative combinations',col]].to_numpy()
    arr = np.insert(arr,obj=0,values=np.array([0,0]),axis=0) #add 0,0 point
    assert sorted(arr[:,0] == arr[:,0]) #check if list is sorted (needed for interpolation function)
    interpolated = np.interp(x=percentages,xp=arr[:,0],fp=arr[:,1])
    return interpolated

def process_no_detour(df):
    """
    Further processes the output of main(), to prepare for plotting the no_detour results
    Merges the many samples per #AoIs to one metric per #AoIs

    no_detour is where no route between Origin and Destination exists during the disruption

    Arguments:
        *df* (DataFrame) : the output of main()

    Returns:
        *no_dt_abs* (DataFrame) : results grouped by combi of absolute AOI AND country
        *no_dt_rel* (DataFrame) : results grouped by combi of relative AOI and country
        """
    # Groups unique combinations of #AoI combination and country
    #group_operations = group_operations()
    no_dt_abs = df.groupby(['AoI combinations', 'country'])['no detour'].agg(group_operations).reset_index()
    no_dt_rel = df.groupby(['AoI relative combinations', 'country'])['no detour'].agg(group_operations).reset_index()

    print(' -------- Process_no_detour() finished --------')
    return(no_dt_abs,no_dt_rel)

def process_extra_time(df):
    """
    Further processes the output of main(), to prepare for plotting the avg extra time results
    Merges the many samples per #AoIs to one metric per #AoIs

    average extra time is the average (of all NUTS-pairs) additional travel time of OD-pairs over te network

    Arguments:
        *df* (DataFrame) : the output of main()

    Returns:
        *extra_time_abs* (DataFrame) : results grouped by combi of absolute AOI AND country
        *extra_time_rel* (DataFrame) : results grouped by combi of relative AOI and country
        """
    # Groups unique combinations of #AoI combination and country
    #group_operations = group_operations()
    extra_time_abs = df.groupby(['AoI combinations', 'country'])['avg extra time'].agg(group_operations).reset_index()
    extra_time_rel = df.groupby(['AoI relative combinations', 'country'])['avg extra time'].agg(group_operations).reset_index()

    print(' -------- Process_extra_time() finished --------')
    return(extra_time_abs,extra_time_rel)

def calc_total_extra_time(df,countries=[],N2=[]):
    """
    Calculates the total extra travel over the time of the network
    Note that that routes for with no detour do not contribute to total travel time

    Arguments:
        *df* (DataFrame) : raw percolation output, the output of main()
        *countries* (string or list of strings) : list with full country names, if not provided: will do all countries
        *N2* (list of strings) : list of country N0 codes: countries for which analysis is run on N2 instead of N3

    Return:
         *df_totaltraveltime* (DataFrame) : raw df with extra columns 'nodetour_routes',
                                            'disrupted_routes', 'sum_extra_travel_time'

    """
    print('Starting total extra travel time calculation, calc_total_extra_time()')
    if isinstance(countries,str):
        countries = [countries]

    from Europe_utils import country_code_from_name
    from math import isclose
    analysis_file = config['paths']['data'] / 'Overview_analysis_2021_5_5.xls'
    file = pd.read_excel(analysis_file,skiprows=0,index_col=1,header=1)
    routes = file[['N2_nr_routes','N3_nr_routes']]
    ### TODO: READING THE THEORETIC NUMBER OF ROUTES IS NOT ALWAYS GOOD, THERE ARE DISCPRECANCIES WITH ACTUAL ROUTES

    #Add empty columns
    cols = ['nodetour_routes','disrupted_routes','sum_extra_travel_time']
    for col in cols:
        df[col] = None
        df[col] = df[col].apply(pd.to_numeric)

    Warnings = [] #countries for which a warning needs to be raised

    #Handle per country
    if len(countries) == 0:
        ac = df.country.unique()  # available countries
    else:
        ac = countries
    for country in ac:
        c = country_code_from_name(country)

        df_sel = df.loc[df['country']==country]
        df = df.loc[df['country']!=country] #remainder

        actual_nr_routes = check_actual_routes() #Todo: dirty fix to enable dashboard because this is not run outside the __main__

        if not c in N2: #calculatoin on NUTS-3
            #nr_routes = routes.at[c,'N3_nr_routes'] #depreciated, beter to derive from the preproc results
            nr_routes = actual_nr_routes[c]
        elif c in N2: #calculation on NUTS-2
            #nr_routes = routes.at[c, 'N2_nr_routes']
            nr_routes = actual_nr_routes[c]

        print(country,c,nr_routes)
        df_sel['disrupted_routes'] = df_sel['disrupted'] * nr_routes / 100
        #pdb.set_trace()
        are_close = [isclose(n_routes, round(n_routes), abs_tol=0.0001) for
                     n_routes in df_sel['disrupted_routes'].unique()]
        if False in are_close: #If any of the value is not close to an integer
            #multiplication % with number of routes should result in whole number
            Warnings.append(country)

        df_sel['nodetour_routes'] = df_sel['no detour'] * nr_routes / 100

        df_sel['sum_extra_travel_time'] = \
            df_sel['avg extra time'] * (df_sel['disrupted_routes'] - df_sel['nodetour_routes'])
        df = df.append(df_sel)

    df_totaltraveltime = df.loc[df['country'].isin(ac)]
    print('calc_total_extra_time() finished!')
    if not len(Warnings) == 0:
        warn("""Absolute number of disrupted routes containes an unexpected value for 
                    {} or
                    {}
                    maybe you choose the wrong NUTS-level (can be N3 or N2) or
                    maybe some route combinations were not sampled
                    """.format(Warnings,[country_code_from_name(country) for country in Warnings]))

    return df_totaltraveltime

def process_total_extra_time(df):
    """
    Further processes the output of calc_total_extra_time
    #Todo: precisely describe what comes out of this function
     - in: per country, per unqiue aoi-comb; the total extra travel time over the network (excluding the no_detour trips
           that cannot take place, because O/D are not longer connected)
     - We aggregate by nr of aoI combinatoins, and calculate summary statistics; in a relative and absolute way
        (compared to #aois)

    Arguments:
        *df* (DataFrame) : df_totaltraveltime, output of calc_total_extra_time

    Returns:
        *total_extra_time_abs* (DataFrame) : results grouped by combi of absolute AOI AND country
        *total_extra_time_rel* (DataFrame) : results grouped by combi of relative AOI and country
        """
    # Groups unique combinations of #AoI combination and country
    #group_operations = group_operations()
    total_extra_time_abs = df.groupby(['AoI combinations', 'country'])['sum_extra_travel_time'].agg(group_operations).reset_index()
    total_extra_time_rel = df.groupby(['AoI relative combinations', 'country'])['sum_extra_travel_time'].agg(group_operations).reset_index()

    print(' -------- Process_total_extra_time() finished --------')
    return(total_extra_time_abs,total_extra_time_rel)

def percolation_summary(df):
    """
    Returns a summary of the percolation results
    
    Arguments:
        *df* (DataFrame) : output of main()
        
    Returns:
        *perc_overview* : overview of percolation analysis
            index = countries
            cols:
                'AoIs' : list of integers representing the number of AoIs (combinations) that has sampled
                'Reps_unique' : observed number of repetitions per combination
                'Mode_reps' : most frequent number of repetition
                'Frequency_mode_reps' : how often the mode is observed
    """
    #Iterate over countries
    #List unique AoI combinations
    #Min_reps; Max_reps; Mode_reps; #Number of AoIs with mode reps
    countries = df.country.unique()
    columns = ['AoIs','Reps_unique','Mode_reps','Frequency_mode_reps']
    perc_overview = pd.DataFrame(index=countries,columns=columns)
    po = perc_overview
    for c in countries:
        group_count = df.loc[df['country']==c].groupby('AoI combinations').count()
        po.at[c,'AoIs'] = list(group_count.index)
        po.at[c,'Reps_unique'] = group_count['disrupted'].unique()
        mode = group_count['disrupted'].mode()
        po.at[c,'Mode_reps'] = mode.values[0]
        po.at[c,'Frequency_mode_reps'] = group_count['disrupted'].value_counts()[mode].values[0]

    return perc_overview



############################################# START PLOTTING FUNCTIONALITY ###############################################

def boxplots_multiple_countries_v1(df,save=False):
    """ Frederique line 88-102
    # Todo: this function should not be used for multiple countries at the same time.

    Create boxplots of multiple countries

    Arguments:
        df (DataFrame) : contains the df of all countries, all combinations
        save (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Returns:
        fig, axes
    """
    df.sort_values('AoI combinations', inplace=True)
    positions = df['AoI combinations'].unique()

    vis1 = df[df['AoI combinations'] <= 5]
    vis2 = df[(df['AoI combinations'] >= 10) & (df['AoI combinations'] <= 50)]
    vis3 = df[df['AoI combinations'] > 50]

    # plot as boxplot - v1
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12, 5))
    p1 = vis1.boxplot(ax=ax1, by='AoI combinations', column='disrupted')
    p2 = vis2.boxplot(ax=ax2, by='AoI combinations', column='disrupted')
    p3 = vis3.boxplot(ax=ax3, by='AoI combinations', column='disrupted')
    ax2.set_title("% routes between NUTS-3 regions disrupted")
    ax1.set_title("")
    ax1.set_ylabel("% preferred routes disrupted")
    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax3.set_title("")
    #fig.suptitle("")

    if save:
        output_images = load_config(config_file)['paths']['output_images']
        plt.savefig(output_images / 'country_comparison_error.png')

    return fig, (ax1,ax2,ax3)

#TODO: skipped second version of the boxplots (Frederique line 104-112)

def plotly_plot(df,countries):
    """
    Fred. line 114-130

    Creates two interactive plotly plots

    Arguments:
        *df* (DataFrame) : contains merged results from all_combinations.csv
        *countries* (list) : list of country names (strings) to plot e.g. ['Albania', 'Belgium']

    Effect:
        Writes to the "output_images" path (see config) the html plots.

    """
    import plotly.express as px
    from plotly.offline import plot

    max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()

    for cntr in countries:
        df.loc[df['country'] == cntr.capitalize(), 'AoI relative combinations'] = \
            df.loc[df['country'] == cntr.capitalize(), "AoI combinations"] / \
                            max_aoi_comb[cntr.capitalize()] * 100

    fig = px.box(df, x="AoI relative combinations", y="disrupted", color="country")
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    fig.update_xaxes(
        title_text="% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country")
    fig.update_yaxes(title_text='% optimal routes disrupted')

    save_figs = load_config(config_file)['paths']['output_images']
    plot(fig, filename=(str(save_figs / 'country_comparison_relative.html')))

    fig = px.box(df.loc[df['AoI combinations'] < 10], x="AoI combinations", y="disrupted", color="country")
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    fig.update_xaxes(title_text="Number of combinations of micro-floods (AoI's)")
    fig.update_yaxes(title_text='% optimal routes disrupted')

    plot(fig, filename=(str(save_figs / 'country_comparions_absolute.html')))

    #Todo: on y-axis, show absolute values on the y-axis
    #Todo: IN the absolute plot: show more values

def boxplot_one_country(df,country,AoIs= 'All',save=False,fig=None,ax=None,**kwargs):
    """
    Fred. line 133-139

    Creates a boxplot of a single country

    Arguments:
        *df* (DataFrame) : contains merged results from all_combinations.csv
        *countries* (string) : (string) name of country to plot
        *positions* (list) : e.g. [1,2,3,4] # nr of AoIs to make boxplots for, default 'All'
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']
        *kwargs* : additional keyword arguments will be passed to df.boxplot()


    Effect:
        Shows the plot
        Can write to the "output_images" path (see config) the plot

    """
    df = df.loc[df['country'] == country]
    #if AoIs != 'All':  df2 = df2.loc[df2['AoI combinations'].isin(AoIs)]

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(by='AoI combinations', column='disrupted', ax=ax,grid=False,**kwargs)
    ax.set_xlabel("Number of combinations of flood events (AoI)")
    ax.set_ylabel("% preferred routes disrupted")
    ax.set_title("% routes between NUTS-3 regions in {} disrupted".format(country))

    if save:
        save_figs = load_config(config_file)['paths']['output_images'] / 'disrupted'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "disrupted_boxplot_{}.png".format(country)
        fig.savefig(save_figs / filename)

    return fig,ax

def aggregated_lineplot_new(df_agg,countries,fill_between=('min','max'),save=False,fig=None,ax=None,clrs='default'):
    """
    Creates an aggregates lineplot for multiple countries

    Arguments:
        *df_agg* (DataFrame) : contains the aggregated results, either relative (df_rel) or absolute (df_abs)
        *countries* (list) : list of strings with names of countries to plot
        *fill_between* (tuple) : indicates which percentiles to feel between
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Returns:
        fig,ax
    """
    #assert fill_between in cols.

    if 'AoI relative combinations' in df_agg.columns: #INDICATES THAT THESE ARE RELATIVE RESULTS
        grouper = 'AoI relative combinations'
        xlabel = "% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country"
        relative = True #need for plotting

    elif 'AoI combinations' in df_agg.columns: #ABSOLUTE RESULTS
        grouper = 'AoI combinations'
        xlabel = "Number of combinations of micro-floods (AoI's)"
        relative = False

    if clrs == 'default':
        clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']
    
    if (fig==None and ax==None): #if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(8, 6))

    lines = df_agg
    for cntry, cl in zip(countries, clrs):
        c = cntry.capitalize()
        ax.plot(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, 'mean'],
                 color=cl, label=c)
        ax.fill_between(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, fill_between[0]],
                         lines.loc[lines['country'] == c, fill_between[1]], alpha=0.3, edgecolor=cl, facecolor=cl, linewidth=0)
    ax.legend()
    ax.set_ylabel("% optimal routes disrupted")
    ax.set_xlabel(xlabel)

    #Todo: add function to link country names with official codes NUTS0

    if save: #TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'aggregate_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "aggregateline_{}_{}.png".format('-'.join(countries),fill_between[0] + '-' + fill_between[1])
        if relative: filename = "aggregateline_{}_{}_relative.png".format(\
                                                        '-'.join(countries),fill_between[0] + '-' + fill_between[1])
        fig.savefig(save_figs / filename)

    return fig,ax

def calculate_metrics(df_rel):
    """
    Calculate indicative metrics per country, to summarize the results of the plots

    Arguments:
        *df_rel* (DataFrame) : results grouped by combi of relative AOI and country

    Returns:
        *df_metrics* : rows countries, cols metrics
    """

    y_col = 'mean'  # can be 'min', 'q_25', 'mean', 'q_75', 'max'

    #Indicator 1: #percentage of AoI at which 20% (?) of the OD-pairs [in the mean line]
    threshold = 30 #%

    #Indicator 2: slope

    print('Start calculating metrics with calculate_metrics()')
    results_columns = ['M1_threshold{:.2f}_{}'.format(threshold,y_col)]
    countries = df_rel.country.unique()
    df_metrics = pd.DataFrame(index=countries,columns=results_columns)
    x_col = df_rel.columns[0] #'AoI relative combinations'

    for c in countries:
        #Metric 1
        metric_name = results_columns[0]
        df = df_rel.loc[df_rel['country'] == c][[x_col,y_col]]
        arr = df.to_numpy().T
        M1 = np.interp(threshold,arr[1],arr[0])
        df_metrics.at[c,metric_name] = M1

    return df_metrics

def no_detour_boxplot(df,country,save=False,fig=None, ax=None,**kwargs):
    """
    Creates boxplot of %OD-pairs with no detour for one country

    Frederique: line 133-141

    Arguments:
        *df* (DataFrame) : raw model results
        *country* (string) : Country name e.g. 'Albania'
        *save* (boolean) : should the file be saved in the folder config['paths']['output_images']
        *fig,ax* (matplotlib) : fig and ax to add the figure to; make new if not provided        
        *kwargs* : additional keyword arguments will be passed to df.boxplot()

    """
    df = df.loc[df['country'] == country]

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(by='AoI combinations', column='no detour', grid=False, ax=ax,**kwargs)
    ax.set_xlabel("Number of combinations of flood events (AoI)")
    ax.set_ylabel("% No detour")
    ax.set_title("% routes between NUTS-3 regions in {} without detour".format(country))

    #Todo (possible): give fig ax as args; enable saving possiblity

    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'no_detour_boxplot'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "noDT_boxplot_{}.png".format(country)
        fig.savefig(save_figs / filename)

    return fig,ax

def no_detour_aggregated_lineplot(no_dt_, countries, fill_between=('min', 'max'), save=False, fig=None, ax=None,clrs='default'):
    """
    Creates an aggregateted lineplot of routes with no detour for multiple countries
    (Function is almost the same as aggregated_lineplot)

    Arguments:
        *no_dt_* (DataFrame) : contains the aggregated results, either relative (no_dt_rel) or absolute (no_dt_abs)
        *countries* (list) : list of strings with names of countries to plot
        *fill_between* (tuple) : indicates which percentiles to feel between
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Returns:
        fig,ax
    """
    # assert fill_between in cols.

    #FIRST DETERMINE IF RELATIVE OR ABSOLUTE RESULTS WERE PROVIDED
    if 'AoI relative combinations' in no_dt_.columns:  # INDICATES THAT THESE ARE RELATIVE RESULTS
        grouper = 'AoI relative combinations'
        xlabel = "% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country"
        relative = True  # needed for plotting

    elif 'AoI combinations' in no_dt_.columns:  # ABSOLUTE RESULTS
        grouper = 'AoI combinations'
        xlabel = "Number of combinations of micro-floods (AoI's)"
        relative = False

    if clrs == 'default':
        clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(8, 6))

    lines = no_dt_
    for cntry, cl in zip(countries, clrs):
        c = cntry.capitalize()
        ax.plot(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, 'mean'],
                color=cl, label=c)
        ax.fill_between(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, fill_between[0]],
                        lines.loc[lines['country'] == c, fill_between[1]], alpha=0.3, edgecolor=cl, facecolor=cl,
                        linewidth=0)
    ax.legend()
    ax.set_ylabel("% routes without detour")
    ax.set_xlabel(xlabel)


    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'no_detour_aggregated_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "noDT_aggregateline_{}_{}.png".format('-'.join(countries), fill_between[0] + '-' + fill_between[1])
        if relative: filename = "noDT_aggregateline_{}_{}_relative.png".format( \
            '-'.join(countries), fill_between[0] + '-' + fill_between[1])
        fig.savefig(save_figs / filename)

    return fig, ax

def extra_time_boxplot(df,country,unit='hr',save=False,fig=None, ax=None,**kwargs):
    """
    Creates boxplot of average (of all detour times) additional detour time per # AoIs

    Frederique: line 214-223

    Arguments:
        *df* (DataFrame) : raw model results
        *unit* (string) : unit of the desired output results, can be 'sec', 'min' or 'hr'
        *country* (string) : Country name e.g. 'Albania'
        *save* (boolean) : should the file be saved in the folder config['paths']['output_images']
        *fig,ax* (matplotlib) : fig and ax to add the figure to; make new if not provided
        *kwargs* : additional keyword arguments will be passed to df.boxplot()


    """


    #Todo: check units; this currently does not work
    df = df.loc[df['country'] == country]

    df['extra_time_min'] = df['avg extra time'] / 60
    df['extra_time_hr'] = df['extra_time_min'] / 60

    if unit == 'sec': col='avg extra time'
    elif unit == 'min' : col='extra_time_min'
    elif unit == 'hr' : col = 'extra_time_hr'
    else: raise ValueError("Unit must be 'sec','min' or 'hr' ")

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(by='AoI combinations', column=[col], ax=ax,grid=False,**kwargs)
    ax.set_xlabel("Number of combinations of flood events (AoI)")
    ax.set_ylabel("average extra time [{}]".format(unit))
    ax.set_title("Average additional time for disrupted routes between NUTS-regions in {}".format(country))


    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'extra_time_boxplot'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "extratime_boxplot_{}.png".format(country)
        fig.savefig(save_figs / filename)

    return fig,ax

def extra_time_aggregated_lineplot(extra_time_, countries, fill_between=('min', 'max'),
                                   save=False, fig=None, ax=None,clrs='default'):
    """
    Creates an aggregateted lineplot of routes with no detour for multiple countries
    (Function is almost the same as aggregated_lineplot)

    Arguments:
        *extra_time_* (DataFrame) : contains the aggregated results, either relative (extra_time_rel) or absolute (extra_time_abs)
        *countries* (list) : list of strings with names of countries to plot
        *fill_between* (tuple) : indicates which percentiles to feel between
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Returns:
        fig,ax
    """
    # assert fill_between in cols.

    #FIRST DETERMINE IF RELATIVE OR ABSOLUTE RESULTS WERE PROVIDED
    if 'AoI relative combinations' in extra_time_.columns:  # INDICATES THAT THESE ARE RELATIVE RESULTS
        grouper = 'AoI relative combinations'
        xlabel = "% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country"
        relative = True  # needed for plotting

    elif 'AoI combinations' in extra_time_.columns:  # ABSOLUTE RESULTS
        grouper = 'AoI combinations'
        xlabel = "Number of combinations of micro-floods (AoI's)"
        relative = False

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(8, 6))

    if clrs == 'default':
        clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']

    lines = extra_time_
    for cntry, cl in zip(countries, clrs):
        c = cntry.capitalize()
        ax.plot(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, 'mean'],
                color=cl, label=c)
        ax.fill_between(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, fill_between[0]],
                        lines.loc[lines['country'] == c, fill_between[1]], alpha=0.3, edgecolor=cl, facecolor=cl,
                        linewidth=0)
    ax.legend()
    ax.set_ylabel("Average extra travel time")
    ax.set_xlabel(xlabel)


    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'extra_time_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "extra_time_aggregateline_{}_{}.png".format('-'.join(countries), fill_between[0] + '-' + fill_between[1])
        if relative: filename = "extra_time_aggregateline_{}_{}_relative.png".format( \
            '-'.join(countries), fill_between[0] + '-' + fill_between[1])
        fig.savefig(save_figs / filename)

    return fig, ax

def total_extra_time_aggregated_lineplot(extra_time_, countries, fill_between=('min', 'max'), save=False, fig=None, ax=None,clrs='default'):
    """
    Creates an aggregateted lineplot of total extra time per disruption in an aggregated lineplot
    (Function is almost the same as aggregated_lineplot)

    Arguments:
        *extra_time_* (DataFrame) : contains the aggregated results, either relative (extra_time_rel) or absolute (extra_time_abs)
        *countries* (list) : list of strings with names of countries to plot
        *fill_between* (tuple) : indicates which percentiles to feel between
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Returns:
        fig,ax
    """
    # assert fill_between in cols.

    #FIRST DETERMINE IF RELATIVE OR ABSOLUTE RESULTS WERE PROVIDED
    if 'AoI relative combinations' in extra_time_.columns:  # INDICATES THAT THESE ARE RELATIVE RESULTS
        grouper = 'AoI relative combinations'
        xlabel = "% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country"
        relative = True  # needed for plotting

    elif 'AoI combinations' in extra_time_.columns:  # ABSOLUTE RESULTS
        grouper = 'AoI combinations'
        xlabel = "Number of combinations of micro-floods (AoI's)"
        relative = False

    if (fig == None and ax == None):  # if No axes and no figure is provided
        fig, ax = plt.subplots(figsize=(8, 6))

    if clrs == 'default':
        clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']

    lines = extra_time_
    for cntry, cl in zip(countries, clrs):
        c = cntry.capitalize()
        ax.plot(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, 'mean'],
                color=cl, label=c)
        ax.fill_between(lines.loc[lines['country'] == c, grouper], lines.loc[lines['country'] == c, fill_between[0]],
                        lines.loc[lines['country'] == c, fill_between[1]], alpha=0.3, edgecolor=cl, facecolor=cl,
                        linewidth=0)
    ax.legend()
    ax.set_ylabel("Average extra travel time")
    ax.set_xlabel(xlabel)


    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config(config_file)['paths']['output_images'] / 'total_extra_time_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "total_extra_time_aggregateline_{}_{}.png".format('-'.join(countries),
                                                                     fill_between[0] + '-' + fill_between[1])
        if relative: filename = "total_extra_time_aggregateline_{}_{}_relative.png".format( \
            '-'.join(countries), fill_between[0] + '-' + fill_between[1])
        fig.savefig(save_figs / filename)

    return fig, ax

def check_actual_routes():
    """"Compares the number of routes in the results excel and the actual routes

    evt filteren op countries
    """
    from math import isnan
    from Europe_utils import country_code_from_name, country_names

    #Load the analysis file
    analysis_file = config['paths']['data'] / 'Overview_analysis_2021_5_5.xls'
    file = pd.read_excel(analysis_file, skiprows=0, index_col=1, header=1)
    routes = file[['N2_nr_routes', 'N3_nr_routes']]

    all_countries = list(routes.index)

    # load the preproc_folder
    preproc_folder = config['paths']['preproc_output']

    warn_no_optimal_routes = []
    actual_routes_dict = {} #key country, value is number of actual routes

    #iterate over all countries in the analysis file
    for i, c in enumerate(all_countries):
        if (not isinstance(c,str)):
            if isnan(c): #empty row in the excel sheet
                continue
        country = country_names(c)
        #print(i,c,country)

        #check if the preproc results file exists for this country
        opt_routes_path = preproc_folder / country / 'optimal_routes_time_{}.feather'.format(country)
        if not opt_routes_path.exists():
            warn_no_optimal_routes.append(c)
            actual_routes_dict[c] = None
            continue
        optimal_routes = pd.read_feather(opt_routes_path)
        #count_origins = len(optimal_routes['origin'].unique())
        #count_destinations = len(optimal_routes['destination'].unique())
        #if not count_origins == count_destinations:
        #    warn('Number of origins {} does not equal number of destiations {}'.format(
        #              count_destinations,count_origins))
        ODs_unique = len(list(set(list(optimal_routes['origin'].unique()) +
                                  list(optimal_routes['destination'].unique()))))

        #Todo: first need to compare the ODS in this list with all available ODS
        theoretical_routes = int(ODs_unique*(ODs_unique-1)*0.5)
        actual_routes = optimal_routes.shape[0]
        actual_routes_dict[c] = actual_routes
        if not theoretical_routes == actual_routes:
            warn('The number of actual routes {} deviates from the expected value {} for {}'.format(theoretical_routes,
                                                            actual_routes,c))

    print('Optimal routes path do not exist for: {}'.format(warn_no_optimal_routes))

    return actual_routes_dict



if __name__ == '__main__':

    #Load configuration file
    config_file = 'config.json' #also declared at the top of the file (ugly fix)
    config = load_config(config_file)

    #Derive how actual preferred routes were calculated in the preprocessing
    actual_nr_routes = check_actual_routes()

    df,df_abs,df_rel = main(config)
    ac = df.country.unique() #available countries

    #fig, axes = boxplots_multiple_countries_v1(df,True)

    #plotly_plot(df,['Albania','Austria','Belgium'])

    boxplot_one_country(df, 'Netherlands',save=True)

    #Sort countries by nr of AoIs
    max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()
    sort_country_by_aoi = sorted(max_aoi_comb.items(), key=lambda x: x[1], reverse=True)
    ac,maxaois = zip(*sort_country_by_aoi) #unzip and replace all country (ac) list
    groups_version1 = [ac[0:4], ac[4:8], ac[8:12], ac[12:16], ac[16:20], ac[20:24], ac[24:28]]

    #Save grouping as pickle
    #import pickle
    #dest = config['paths']['data'] / 'groups' / 'group_4_by_nr_AoI.p'
    #with open(dest,'wb') as f:
    #    pickle.dump(groups_version1,f)

    #Group by country area
    #Load second group version
    import json
    group_file = config['paths']['data'] / 'groups' / '6groupsof6_byactivesize.json'
    with open(group_file, 'rb') as f:
        groups = json.load(f)
    groups = groups['groups']

    fill_between=('q_05','q_95')

    ### Make aggregated lineplots
    #for countries in groups_version1:
         #you can change the df_rel and df_abs here; and the fill_between=('q_25','q_75') or ('min','max')
         #aggregated_lineplot_new(df_abs,countries,fill_between=fill_between,save=True) #

    for group in groups.values():
        countries = eu.country_names(group)
        aggregated_lineplot_new(df_rel,countries,fill_between=fill_between,save=True)

    # countries = ['Portugal','Slovakia']
    # aggregated_lineplot_new(df_rel, countries, save=True)  #

    ### CALCULATE SOME KEY METRICS FROM THE PERCOLATION RESULTS ###
    df_metrics = calculate_metrics(df_rel)

    ### PROCESS NO DETOUR RESULTS ###
    no_dt_abs, no_dt_rel = process_no_detour(df)
    no_detour_boxplot(df, 'Netherlands', True)
    extra_time_boxplot(df, 'Hungary',unit='min', save=True)
    #Example of no detour results plotting
    #no_detour_aggregated_lineplot(no_dt_rel, ['Germany', 'France'])
    #for countries in groups_version1:
         #you can change the df_rel and df_abs here; and the fill_between=('q_25','q_75') or ('min','max')
         #no_detour_aggregated_lineplot(no_dt_abs,countries=countries,fill_between=fill_between,save=True) #

    #Create no detour boxplots per country
    #for country in ac:
    #    no_detour_boxplot(df, country, True)

    ### START EXTRA TIME VISUALISATIONS ###
    # plot as boxplot - avg extra time
    #for country in ac:
    #    extra_time_boxplot(df, country, save=True)

    #extra_time_abs,extra_time_rel = process_extra_time(df)
    #for countries in groups_version1:
    #you can change the df_rel and df_abs here; and the fill_between=('q_25','q_75') or ('min','max')
        #extra_time_aggregated_lineplot(extra_time_abs,fill_between=fill_between, countries=countries,save=True)

    ### START AGGREGATED EXTRA TIME VISUALISATIONS ###
    countries = groups_version1[-1]
    NUTS2_analysis = ['DE', 'BE', 'NL'] #countries run on N2 instead of N3
    for countries in groups_version1: #process results per group of countries
        df_totaltraveltime = calc_total_extra_time(df,countries=countries,N2=NUTS2_analysis)
        total_extra_time_abs, total_extra_time_rel = process_total_extra_time(df_totaltraveltime)
        total_extra_time_aggregated_lineplot(total_extra_time_abs,countries,fill_between=fill_between,save=True)


    print('end of script')



