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

from utils import load_config

def combine_finished_stochastic(finished_folder):
    """Combines the separate csv files create by the parallel processed stochastic results.
    Args:
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
            df = pd.read_csv(files[0])
            for file in files[1:]:
                df_add = pd.read_csv(file)
                df = pd.concat([df, df_add], sort='False')
            df.to_csv(finished_folder / "aoi_{}.csv".format(folder.stem))

    print('Combine_finished_stochastic finished for {}'.format(finished_folder))

def df_stochastic_results(folder):
    """
    Summarizes the different combinations in a dataframe for the country

    Arguments:
        *country_folder* (Path) : finished results of a country, e.g.
                    country_results_folder/albania/finished

    Returns:
        *df* (DataFrame) : contains the results per #combinations (rows)

    """
    files = [f for f in os.listdir(folder) if (os.path.isfile(os.path.join(folder, f))) and (f.endswith('.csv'))]

    # load data
    df = pd.DataFrame(columns=['AoI combinations', 'disrupted', 'avg extra time', 'AoI removed', 'no detour'])
    for f in files:
        df_new = pd.read_csv(os.path.join(folder, f))
        df = pd.concat([df, df_new], ignore_index=True, sort=False)

    df['AoI combinations'] = df['AoI combinations'].astype(int)
    return df

# Everythin below  is a sort preprocessing step, before the actual plotting starts
#if __name__ == '__main__':
if False:
    config = load_config()
    country_results_folder = config['paths']['country_results']
    #Example folder structure: country_results_folder/albania/finished/

    #Step 1: for each country, summarize individual samples to results per combination
    country = 'austria'
    finished_folder = country_results_folder / country / 'finished'
    #combine_finished_stochastic(finished_folder)

    #Step 2: for each country, summarize combinations in a dataframe
    countries = ['albania','austria','belgium']
    folders = [df_stochastic_results(folder=country_results_folder / c / 'finished') for c in countries]
    dict_dfs = dict(zip(countries, folders)) #keys are countries, values dataframes with results

    #Step 3: Summarize the results of all countries
    #folder_results = r'D:\COACCH_paper\data\output\{}'
    folder_results = config['paths']['output_data']
    # group the dataframes
    for c in countries:
        temp_df = dict_dfs[c]
        temp_df['country'] = c.capitalize()
        dict_dfs[c] = temp_df

    df = pd.concat(list(dict_dfs.values()), sort=False)
    df.to_csv(folder_results / 'all_combinations.csv')

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
    fig.suptitle("")

    if save:
        output_images = load_config()['paths']['output_images']
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

    save_figs = load_config()['paths']['output_images']
    plot(fig, filename=(str(save_figs / 'country_comparison_relative.html')))

    fig = px.box(df.loc[df['AoI combinations'] < 10], x="AoI combinations", y="disrupted", color="country")
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    fig.update_xaxes(title_text="Number of combinations of micro-floods (AoI's)")
    fig.update_yaxes(title_text='% optimal routes disrupted')

    plot(fig, filename=(str(save_figs / 'country_comparions_absolute.html')))

    #Todo: on y-axis, show absolute values on the y-axis
    #Todo: IN the absolute plot: show more values

def boxplot_one_country(df,country,AoIs= 'All',save=False):
    """
    Fred. line 133-139

    Creates a boxplot of a single country

    Arguments:
        *df* (DataFrame) : contains merged results from all_combinations.csv
        *countries* (string) : (string) name of country to plot
        *positions* (list) : e.g. [1,2,3,4] # nr of AoIs to make boxplots for, default 'All'
        *save* (Boolean) : should the file be saved in the folder config['paths']['output_images']

    Effect:
        Shows the plot
        Can write to the "output_images" path (see config) the plot

    """
    df2 = df.copy()
    df2.loc[df2['country'] == country]
    if AoIs != 'All':  df2 = df2.loc[df2['AoI combinations'].isin(AoIs)]

    df2.boxplot(by='AoI combinations', column='disrupted')
    plt.xlabel("Number of combinations of flood events (AoI)")
    plt.ylabel("% preferred routes disrupted")
    plt.title("% routes between NUTS-3 regions in {} disrupted".format(country))
    plt.suptitle("")
    plt.show()

    if save:
        save_figs = load_config()['paths']['output_images']
        filename = "boxplot_{}.pdf".format(country)
        plt.savefig(save_figs / filename)

    if save: #TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config()['paths']['output_images'] / 'aggregate_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "aggregateline_{}_{}.png".format('-'.join(countries),fill_between[0] + '-' + fill_between[1])
        if relative: filename = "aggregateline_{}_{}_relative.png".format(\
                                                        '-'.join(countries),fill_between[0] + '-' + fill_between[1])
        plt.savefig(save_figs / filename)

    return fig,ax

def aggregated_lineplot_new(df_agg,countries,fill_between=('min','max'),save=False,fig=None,ax=None):
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
    fig.show()

    #Todo: add function to link country names with official codes NUTS0

    if save: #TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config()['paths']['output_images'] / 'aggregate_line'
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

def no_detour_boxplot(df,country,save=False):
    """
    Creates boxplot of %OD-pairs with no detour for one country

    Frederique: line 133-141

    Arguments:
        *df* (DataFrame) : raw model results
        *country* (string) : Country name e.g. 'Albania'

    """
    df = df.loc[df['country'] == country]

    df.boxplot(by='AoI combinations', column='disrupted', figsize=(12, 5))
    plt.xlabel("Number of combinations of flood events (AoI)")
    plt.ylabel("% No detour")
    plt.title("% routes between NUTS-3 regions in {} without detour".format(country))
    plt.show()

    #Todo (possible): give fig ax as args; enable saving possiblity

    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config()['paths']['output_images'] / 'no_detour_boxplot'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "noDT_boxplot_{}.png".format(country)
        plt.savefig(save_figs / filename)

    return None

def no_detour_aggregated_lineplot(no_dt_, countries, fill_between=('min', 'max'), save=False, fig=None, ax=None):
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
    fig.show()


    if save:  # TODO REPLACE ALL INSTANCES OF THIS PART OF CODE WITH A SPECIAL FUNCTION
        save_figs = load_config()['paths']['output_images'] / 'no_detour_aggregated_line'
        if not save_figs.exists(): save_figs.mkdir()
        filename = "noDT_aggregateline_{}_{}.png".format('-'.join(countries), fill_between[0] + '-' + fill_between[1])
        if relative: filename = "noDT_aggregateline_{}_{}_relative.png".format( \
            '-'.join(countries), fill_between[0] + '-' + fill_between[1])
        fig.savefig(save_figs / filename)

    return fig, ax


def main(config):
    """
    Reads the (merged) outputs of the percolation analysis, and does some basic assembly work which is useful
    for many visualisations.

    Arguments:
        *config* (dict) : containing the configuration paths

    Returns:
        *df* (DataFrame) : the raw results
        *df_abs* (DataFrame) : results grouped by combi of absolute AOI AND country
        *df_rel* (DataFrame) : results grouped by combi of relative AOI and country
    """
    print(' -------- main() starting --------')
    folder_results = config['paths']['output_data']

    # READ SOURCE FILE
    df = pd.read_csv((folder_results / 'all_combinations.csv'),index_col=0,sep=';')
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'])
    print('Succesfully loaded source file as dataframe, with columns:')
    print(df.columns)
    print('Available for {} countries'.format(len(df.country.unique())))

    available_countries = df.country.unique()

    print('Grouping per AoI-country combination')
    max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()

    for cntr in available_countries :
        df.loc[df['country'] == cntr.capitalize(), 'AoI relative combinations'] = \
            df.loc[df['country'] == cntr.capitalize(), "AoI combinations"] / \
            max_aoi_comb[cntr.capitalize()] * 100

    # Groups unique combinations of #AoI combination and country (stats are about %OD pairs disrupted)
    group_operations = ['min', q_25, 'mean', q_75, 'max']
    df_abs = df.groupby(['AoI combinations', 'country'])['disrupted'].agg(group_operations).reset_index()
    df_rel = df.groupby(['AoI relative combinations', 'country'])['disrupted'].agg(group_operations).reset_index()

    print(' -------- main() finished --------')
    return(df,df_abs,df_rel)

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
    # Groups unique combinations of #AoI combination and country (stats are about %OD pairs disrupted)
    group_operations = ['min', q_25, 'mean', q_75, 'max']
    no_dt_abs = df.groupby(['AoI combinations', 'country'])['no detour'].agg(group_operations).reset_index()
    no_dt_rel = df.groupby(['AoI relative combinations', 'country'])['no detour'].agg(group_operations).reset_index()

    print(' -------- Process_no_detour() finished --------')
    return(no_dt_abs,no_dt_rel)

#Todo: remove the .show statement; or suppress user warning
# UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline,
# which is a non-GUI backend, so cannot show the figure. fig.show()

# n,th Percentile
def q_n(x,n):
    """Returns the nth percentile of a series """
    return x.quantile(n)

def q_25(x):
    return(q_n(x,0.25))

def q_75(x):
    return(q_n(x,0.75))




if __name__ == '__main__':
    #Load configuration file
    config = load_config()

    df,df_abs,df_rel = main(config)
    ac = df.country.unique() #available countries

    #fig, axes = boxplots_multiple_countries_v1(df,True)

    #plotly_plot(df,['Albania','Austria','Belgium'])

    #boxplot_one_country(df, 'Albania')

    print('ho es')

    #Sort countries by nr of AoIs
    max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()
    sort_country_by_aoi = sorted(max_aoi_comb.items(), key=lambda x: x[1], reverse=True)
    ac,maxaois = zip(*sort_country_by_aoi) #unzip and replace all country (ac) list

    ### Make aggregated lineplots
    # for countries in [ac[0:4],ac[4:8],ac[8:12],ac[12:16],ac[16:20],ac[20:22]]:
    #     #you can change the df_rel and df_abs here; and the fill_between=('q_25','q_75') or ('min','max')
    #     aggregated_lineplot_new(df_rel,countries,fill_between=('q_25','q_75'),save=True) #
    #
    # countries = ['Portugal','Slovakia']
    # aggregated_lineplot_new(df_rel, countries, save=True)  #

    ### CALCULATE SOME KEY METRICS FROM THE PERCOLATION RESULTS ###
    df_metrics = calculate_metrics(df_rel)

    ### PROCESS NO DETOUR RESULTS ###
    no_dt_abs, no_dt_rel = process_no_detour(df)
    #Example of no detour results plotting
    #no_detour_aggregated_lineplot(no_dt_rel, ['Germany', 'France'])
    #for countries in [ac[0:4],ac[4:8],ac[8:12],ac[12:16],ac[16:20],ac[20:22]]:
         #you can change the df_rel and df_abs here; and the fill_between=('q_25','q_75') or ('min','max')
    #     no_detour_aggregated_lineplot(no_dt_abs,countries,fill_between=('q_25','q_75'),save=True) #

    #Create no detour boxplots per country
    for country in ac:
        no_detour_boxplot(df, country, True)

    print('end of script')



