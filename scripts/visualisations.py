# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14

@author: Frederique de Groen

Part of a general tool for criticality analysis of networks.
Visualized the connections between origin/destination pairs.


Changelog
Kees (26 jan):


"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

import network_functions as nf
import plotly.express as px
from plotly.offline import plot

#save_figs = r"P:\osm_flood\network_analysis\igraph\figures"


#Is apparantly no longer used? Commented out below.
def combine_finished_stochastic(finished_folder):
    """Combines the separate csv files create by the parallel processed stochastic results.
    Args:
        input_path: path to the folder where the separate csv's are saved
        output_path: output path, full path to the csv that is merged from all csv's

    Returns:
        None
    """
    folders = os.listdir(finished_folder)  # folder correspond to items of the combs_list defined before

    for folder in folders:
        files = os.listdir(os.path.join(finished_folder, folder))
        df = pd.read_csv(os.path.join(finished_folder, folder, files[0]))
        for file in files[1:]:
            df_add = pd.read_csv(os.path.join(finished_folder, folder, file))
            df = pd.concat([df, df_add], sort='False')
        df.to_csv(os.path.join(finished_folder, "aoi_{}.csv".format(folder.split(".")[0])))

print('pause')

def df_stochastic_results(folder):
    finished_folder = os.path.join(folder, 'finished')
    files = [f for f in os.listdir(finished_folder) if (os.path.isfile(os.path.join(finished_folder, f))) and (f.endswith('.csv'))]

    # load data
    df = pd.DataFrame(columns=['AoI combinations', 'disrupted', 'avg extra time', 'AoI removed', 'no detour'])
    for f in files:
        df_new = pd.read_csv(os.path.join(finished_folder, f))
        df = pd.concat([df, df_new], ignore_index=True, sort=False)

    df['AoI combinations'] = df['AoI combinations'].astype(int)
    df.to_csv(os.path.join(folder, "{}.csv".format(folder.split('\\')[-1])))
    return df


## STOCHASTIC ANALYSIS VISUALISATIONS ##
# compile results from parallel processing
countries = ['albania', 'austria', 'belgium', 'bulgaria', 'croatia', 'czechia', 'denmark', 'finland', 'france',
             'germany', 'greece', 'hungary', 'ireland', 'netherlands', 'norway', 'poland', 'portugal', 'romania',
             'serbia', 'slovakia', 'spain', 'switzerland']

for c in countries:
    combine_finished_stochastic(os.path.join(r"P:\osm_flood\network_analysis\igraph", c, 'finished'))

folder_results = r"P:\osm_flood\network_analysis\igraph\{}"
folders = [df_stochastic_results(folder=folder_results.format(c)) for c in countries]
dict_dfs = dict(zip(countries, folders))

# group the dataframes
for c in countries:
    temp_df = dict_dfs[c]
    temp_df['country'] = c.capitalize()
    dict_dfs[c] = temp_df

df = pd.concat(list(dict_dfs.values()), sort=False)
df.to_csv(os.path.join(folder_results.format(''), 'all_combinations.csv'))

df = pd.read_csv(os.path.join(folder_results.format(''), 'all_combinations.csv'))

df.sort_values('AoI combinations', inplace=True)
df.sort_values('disrupted', inplace=True)
positions = df['AoI combinations'].unique()

vis1 = df[df['AoI combinations'] <= 5]
vis2 = df[(df['AoI combinations'] >= 10) & (df['AoI combinations'] <= 50)]
vis3 = df[df['AoI combinations'] > 50]

# plot as boxplot - v1
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12,5))
p1 = vis1.boxplot(ax=ax1, by='AoI combinations', column='disrupted')
p2 = vis2.boxplot(ax=ax2, by='AoI combinations', column='disrupted')
p3 = vis3.boxplot(ax=ax3, by='AoI combinations', column='disrupted')
ax2.set_title("% routes between NUTS-3 regions in Austria disrupted")
ax1.set_title("")
ax1.set_ylabel("% preferred routes disrupted")
ax1.set_xlabel("")
ax3.set_xlabel("")
ax3.set_title("")
plt.suptitle("")
# plt.show()
plt.savefig(os.path.join(save_figs, "preliminary_comparison_AT_AL_IE_SE_BE.png"))
plt.close()

# plot as boxplot - v2
df.boxplot(by='AoI combinations', column='disrupted', figsize=(12, 5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes disrupted")
plt.title("% routes between NUTS-3 regions in Belgium disrupted")
plt.suptitle("")
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\belgium\stochastic_results\figures\belgium_disrupted_boxplot_1000reps_v2.png")
plt.close()

# plotly plot
max_aoi_comb = df.groupby('country')["AoI combinations"].max().to_dict()

for cntr in countries:
    df.loc[df['country'] == cntr.capitalize(), 'AoI relative combinations'] = df.loc[df['country'] == cntr.capitalize(), "AoI combinations"] / max_aoi_comb[cntr.capitalize()] * 100

fig = px.box(df, x="AoI relative combinations", y="disrupted", color="country")
fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
fig.update_xaxes(title_text="% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country")
fig.update_yaxes(title_text='% optimal routes disrupted')
plot(fig, filename=os.path.join(save_figs, 'comparison_AT_AL_IE_IT_SE_BE_relative.html'))

fig = px.box(df, x="AoI combinations", y="disrupted", color="country")
fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
fig.update_xaxes(title_text="Number of combinations of micro-floods (AoI's)")
fig.update_yaxes(title_text='% optimal routes disrupted')
plot(fig, filename=os.path.join(save_figs, 'comparison_all_countries.html'))

# plot as boxplot - v3
positions = [1,2,3,4,5,8,9,10,11,12] #,15,16,17,18,19]
df.boxplot(by='AoI combinations', column='disrupted', positions=positions, figsize=(12,5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes disrupted")
plt.title("% routes between NUTS-3 regions in Austria disrupted")
plt.suptitle("")
# plt.show()
plt.savefig(os.path.join(save_figs, 'albania_disrupted_boxplot_100reps.png'))
plt.close()

df.boxplot(by='AoI combinations', column='disrupted', positions=positions, figsize=(12,5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes disrupted")
plt.title("% routes between NUTS-3 regions in Austria disrupted")
plt.suptitle("")
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_disrupted_boxplot_1000reps.png")
plt.close()


# aggregated lineplot with area for min and max : ABSOLUTE
agg_funcs = dict(Min='min', Mean='mean', Max='max')
lines = df.groupby(['AoI combinations', 'country'])['disrupted'].agg(agg_funcs).reset_index()
# lines = lines.loc[lines['AoI combinations'] <= 200]
clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']
fig, ax = plt.subplots(figsize=(8,6))
for cntry, cl in zip(countries, clrs):
    c = cntry.capitalize()
    plt.plot(lines.loc[lines['country'] == c, 'AoI combinations'], lines.loc[lines['country'] == c, 'Mean'],
             color=cl, label=c)
    plt.fill_between(lines.loc[lines['country'] == c, 'AoI combinations'], lines.loc[lines['country'] == c, 'Min'],
                     lines.loc[lines['country'] == c, 'Max'], alpha=0.3, edgecolor=cl, facecolor=cl, linewidth=0)
plt.legend()
# plt.xlim(0,11)
plt.ylabel("% optimal routes disrupted")
plt.xlabel("Number of combinations of micro-floods (AoI's)")
# plt.xscale('log')
plt.show()
plt.savefig(os.path.join(save_figs, 'aggregated_line_AT_AL_IE_IT_SE_BE.png'), dpi=300, bbox_inches='tight')
plt.close()

# aggregated lineplot with area for min and max : RELATIVE
agg_funcs = dict(Min='min', Mean='mean', Max='max')
lines = df.groupby(['AoI relative combinations', 'country'])['disrupted'].agg(agg_funcs).reset_index()
# lines = lines.loc[lines['AoI relative combinations'] <= 12]
clrs = ['darkblue', 'red', 'green', 'purple', 'orange', 'skyblue']
fig, ax = plt.subplots(figsize=(8,6))
for cntry, cl in zip(countries, clrs):
    c = cntry.capitalize()
    plt.plot(lines.loc[lines['country'] == c, 'AoI relative combinations'], lines.loc[lines['country'] == c, 'Mean'],
             color=cl, label=c)
    plt.fill_between(lines.loc[lines['country'] == c, 'AoI relative combinations'], lines.loc[lines['country'] == c, 'Min'],
                     lines.loc[lines['country'] == c, 'Max'], alpha=0.3, edgecolor=cl, facecolor=cl, linewidth=0)
# plt.xlim(10,21)
# plt.ylim(20,40)
plt.legend()
plt.ylabel("% optimal routes disrupted")
plt.xlabel("% of combinations of micro-floods (AoI's) of the maximum number of micro-floods per country")
# plt.xscale('log')
plt.show()
plt.savefig(os.path.join(save_figs, 'aggregated_line_AT_AL_IE_IT_SE_BE_relative_till20.png'), dpi=300, bbox_inches='tight')
plt.close()


# plot points & line
# calculate the mean, min and max
lines = df.groupby('AoI combinations')['disrupted'].agg(['mean', 'min', 'max']).reset_index()
ax = df.plot.scatter(x='AoI combinations', y='disrupted', color='grey', alpha=0.5)
lines.plot.line(x='AoI combinations', y='mean', ax=ax, markevery=100)
lines.plot.line(x='AoI combinations', y='min', ax=ax, markevery=100)
lines.plot.line(x='AoI combinations', y='max', ax=ax, markevery=100)
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes disrupted")
plt.title("% routes between NUTS-3 regions in Albania disrupted")
plt.suptitle("")
plt.legend()
# plt.show()
plt.savefig(os.path.join(save_figs, "albania_disrupted_line_100reps.png"))
plt.close()

# plot as boxplot - avg extra time
df['extra_time_min'] = df['avg extra time'] / 60
df['extra_time_hr'] = df['extra_time_min'] / 60
df.boxplot(by='AoI combinations', column='extra_time_hr', positions=positions, figsize=(12,5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("average extra time [hours]")
plt.title("extra time for disrupted routes between NUTS-3 regions in Austria")
plt.suptitle("")
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_extra_time_hrs_boxplot_1000reps.png")
plt.close()

df.boxplot(by='AoI combinations', column='extra_time_min', positions=positions, figsize=(12,5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("average extra time [minutes]")
plt.title("extra time for disrupted routes between NUTS-3 regions in Austria")
plt.suptitle("")
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_extra_time_mins_boxplot_1000reps.png")
plt.close()

# plot points & line - avg extra time
# calculate the mean, min and max
lines = df.groupby('AoI combinations')['extra_time_hr'].agg(['mean', 'min', 'max']).reset_index()
ax = df.plot.scatter(x='AoI combinations', y='extra_time_hr', color='grey', alpha=0.5)
# lines.plot.line(x='AoI combinations', y='mean', ax=ax)
# lines.plot.line(x='AoI combinations', y='min', ax=ax)
# lines.plot.line(x='AoI combinations', y='max', ax=ax)
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("average extra time [hours]")
plt.title("extra time for disrupted routes between NUTS-3 regions in Austria")
plt.suptitle("")
# plt.legend()
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_extra_time_hrs_scatter_1000reps.png")
plt.close()

lines = df.groupby('AoI combinations')['extra_time_min'].agg(['mean', 'min', 'max']).reset_index()
ax = df.plot.scatter(x='AoI combinations', y='extra_time_min', color='grey', alpha=0.5)
# lines.plot.line(x='AoI combinations', y='mean', ax=ax)
# lines.plot.line(x='AoI combinations', y='min', ax=ax)
# lines.plot.line(x='AoI combinations', y='max', ax=ax)
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("average extra time [minutes]")
plt.title("extra time for disrupted routes between NUTS-3 regions in Austria")
plt.suptitle("")
# plt.legend()
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_extra_time_mins_scatter_1000reps.png")
plt.close()

# plot as boxplot - no detour
df.boxplot(by='AoI combinations', column='no detour', positions=positions, figsize=(12,5))
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% of preferred routes with no detour possible")
plt.title("no detour possibility for disrupted routes between NUTS-3 regions in Austria")
plt.suptitle("")
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_no_detour_boxplot_1000reps.png")
plt.close()

# plot mean line - disruption comparison
lines = df.groupby(['AoI combinations', 'country']).agg({
    'no detour':'mean',
    'disrupted':'mean'}).reset_index()
ax = lines.loc[(lines['country'] == 'Belgium') & (lines['AoI combinations'] < 200)].plot.line(x='AoI combinations', y='disrupted', label='Belgium')
lines.loc[(lines['country'] == 'Austria') & (lines['AoI combinations'] < 200)].plot.line(x='AoI combinations', y='disrupted', label='Austria', ax=ax)
# lines.plot.line(x='AoI combinations', y='disrupted', ax=ax)
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes disrupted")
plt.title("% routes between NUTS-3 regions in Austria and Belgium disrupted")
plt.suptitle("")
plt.legend()
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\comparison_figures\austria_vs_belgium_mean_line2.png")
plt.close()

# no detour comparison
lines = df.groupby(['AoI combinations', 'country']).agg({
    'no detour':'mean',
    'disrupted':'mean'}).reset_index()
ax = lines.loc[(lines['country'] == 'Belgium') & (lines['AoI combinations'] < 100)].plot.line(x='AoI combinations', y='no detour', label='Belgium')
lines.loc[(lines['country'] == 'Austria') & (lines['AoI combinations'] < 100)].plot.line(x='AoI combinations', y='no detour', label='Austria', ax=ax)
# lines.plot.line(x='AoI combinations', y='disrupted', ax=ax)
plt.xlabel("Number of combinations of flood events (AoI)")
plt.ylabel("% preferred routes with no detour possible")
plt.title("no detour possibility for disrupted routes between\nNUTS-3 regions in Austria and Belgium")
plt.suptitle("")
plt.legend()
plt.show()
plt.savefig(r"P:\osm_flood\network_analysis\comparison_figures\austria_vs_belgium_mean_line_no_detour.png")
plt.close()


# details on outlier in stochastic analysis
aois_removed = df.loc[(df['AoI combinations'] == 1) & (df['disrupted'] == min(df.loc[df['AoI combinations'] == 1, 'disrupted']))]['AoI removed'].iloc[0][1:-1]
aois_removed = [float(x) for x in aois_removed.split(", ")]

for col in df.columns:
    print(df.loc[(df['AoI removed'] == 31974), col])

# 'replay' the removal of the AoI's
G = nx.read_gpickle(
    r'N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\austria_graph_multiple_aoi.gpickle')
pref_time = gpd.read_file(
    r"N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\4_multi_link_one_aoi_time\time_pref_routes.shp")

# initiate variables
df = gpd.GeoDataFrame(columns=['AoI combinations', 'disrupted', 'extra time', 'AoI removed', 'no detour', 'odpair',
                               'geometry'], geometry='geometry', crs={'init': 'epsg:4326'})
tot_routes = len(pref_time.index)
weighing = 'time'

extra_time = []
aoi = [31607, 33343, 33217, 32405, 30389]
to_remove = [(e[0], e[1]) for e in G.edges.data() if (set(e[-1]['AoI_rp100']) & set(aoi))]
H = G.copy()
H.remove_edges_from(to_remove)

for ii in range(len(pref_time.index)):
    o, d = pref_time.iloc[ii][['o_node', 'd_node']]
    o = int(o)
    d = int(d)

    # check if the nodes are still connected
    if nx.has_path(H, o, d):
        # calculate the alternative distance if that edge is unavailable
        alt_route = nx.dijkstra_path_length(H, o, d, weight=weighing)

        # append to list of alternative routes to get the average
        extra_time = alt_route - pref_time.iloc[ii]['time']

        disrupted = 0
        detour = "same"

        if pref_time.iloc[ii]['time'] != alt_route:
            # the alternative route is different from the preferred route
            disrupted = 1
            detour = "alt_route"
    else:
        # append to calculation dataframe
        disrupted = 1
        detour = "no_detour"

    df = df.append({'AoI combinations': 5, 'disrupted': disrupted, 'extra time': extra_time, 'AoI removed': aoi,
                    'no detour': detour, 'odpair': str(pref_time.iloc[ii]['origin'])+str(pref_time.iloc[ii]['destinatio']),
                    'geometry': pref_time.iloc[ii]['geometry']}, ignore_index=True)

nf.gdf_to_shp(df, r"N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\4_multi_link_one_aoi_time\pref_time_aoi_31607_33343_33217_32405_30389.shp")

## HEATMAPS ##
disrupted_dist = gpd.read_file(r"N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\3_multi_link_one_aoi_distance\AoI_rp100_OD_pairs_dist_alt_route.shp")
disrupted_time = gpd.read_file(r"N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\4_multi_link_one_aoi_time\AoI_rp100_OD_pairs_time_alt_route.shp")

disrupted_dist['dif_km'] = disrupted_dist['dif_dist'] / 1000
disrupted_time['dif_min'] = disrupted_time['dif_s'] / 60

dfs = [
    pref_dist[['dist', 'origin', 'destinatio']],
    pref_time[['time_min', 'origin', 'destinatio']],
    disrupted_dist[['dif_km', 'origin', 'destinatio']],
    disrupted_dist[['dif_km', 'origin', 'destinatio']],
    disrupted_time[['dif_min', 'origin', 'destinatio']],
    disrupted_time[['dif_min', 'origin', 'destinatio']]
]
titles = [
    "Distance of preferred routes between NUTS-3 regions in Austria",
    "Duration of preferred routes between NUTS-3 regions in Austria",
    "Average extra distance for disrupted routes between NUTS-3 regions in Austria",
    "Standard deviation of the extra distance(s) for disrupted routes between NUTS-3 regions in Austria",
    "Average extra time for disrupted routes between NUTS-3 regions in Austria",
    "Standard deviation of the extra time for disrupted routes between NUTS-3 regions in Austria"
]
ylabels = [
    'distance [km]',
    'time [minutes]',
    'average extra distance [km]',
    'standard deviation of extra distance (df=0) [km]',
    'average extra time [minutes]',
    'standard deviation of extra time (df=0) [minutes]'
]
save_paths = [
    os.path.join(save_figs, 'heatmap_preferred_routes_dist.png'),
    os.path.join(save_figs, 'heatmap_preferred_routes_time.png'),
    os.path.join(save_figs, 'heatmap_disrupted_routes_dist_avg.png'),
    os.path.join(save_figs, 'heatmap_disrupted_routes_dist_std.png'),
    os.path.join(save_figs, 'heatmap_disrupted_routes_time_avg.png'),
    os.path.join(save_figs, 'heatmap_disrupted_routes_time_std.png')
]

i = 1
nf.plot_heatmap(dfs[i], titles[i], ylabels[i], save_paths[i], aggregate=False, agg_type=None)

# heatmap showing amount of floods influencing preferred routes between NUTS-3 regions
df = disrupted_dist[['dif_km', 'origin', 'destinatio']]
df = df.groupby(['origin', 'destinatio']).size().reset_index()

# duplicate the origin / destination column and change origin for destination and destination for origin, to fill the whole heatmap
df2 = pd.DataFrame({0: df[0], 'origin': df['destinatio'], 'destinatio': df['origin']})
df = pd.concat([df, df2], ignore_index=True, sort=False)
a = df.pivot(index='origin', columns='destinatio', values=0)
a.fillna(value=0, inplace=True)

df = disrupted_time[['dif_min', 'origin', 'destinatio']]
df = df.groupby(['origin', 'destinatio']).size().reset_index()

# duplicate the origin / destination column and change origin for destination and destination for origin, to fill the whole heatmap
df2 = pd.DataFrame({0: df[0], 'origin': df['destinatio'], 'destinatio': df['origin']})
df = pd.concat([df, df2], ignore_index=True, sort=False)
b = df.pivot(index='origin', columns='destinatio', values=0)
b.fillna(value=0, inplace=True)

# difference matrix
a = a - b
max_val = a.max().max()

ylabs = list(a.index)
xlabs = list(a.columns)

fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(a, cmap='RdBu_r', vmin=0, vmax=max_val,
               extent=[0, len(xlabs), 0, len(ylabs)],
               interpolation='none', origin='lower')

# We want to show all ticks...
plt.xticks([x - 0.5 for x in list(range(1, len(xlabs) + 1))], xlabs, rotation=90)
plt.yticks([x - 0.5 for x in list(range(1, len(ylabs) + 1))], ylabs)

# ... and label them with the respective list entries
ax.set_xticklabels(xlabs)
ax.set_yticklabels(ylabs)
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Difference in floods disrupting the preferred route (distance-time)', rotation=90)
# cbar.ax.set_ylabel('Floods disrupting the preferred quickest route', rotation=90)
plt.title('Difference in the amount of floods (AoI\'s) per preferred route measured by distance and time')
# plt.title('Amount of floods (AoI\'s) per preferred quickest route')
plt.show()
fig.savefig(os.path.join(save_figs, 'heatmap_dist_vs_time_floods_per_route.png'))
# fig.savefig(os.path.join(save_figs, 'heatmap_time_floods_per_route.png'))
plt.close()


# heatmap showing connectivity between NUTS-3 regions
df = disrupted_dist[['dif_km', 'origin', 'destinatio']]
df = df.groupby(['origin', 'destinatio']).size().reset_index()
max_val = df[0].max()

# duplicate the origin / destination column and change origin for destination and destination for origin, to fill the whole heatmap
df2 = pd.DataFrame({0: df[0], 'origin': df['destinatio'], 'destinatio': df['origin']})
df = pd.concat([df, df2], ignore_index=True, sort=False)
a = df.pivot(index='origin', columns='destinatio', values=0)
a.fillna(value=0, inplace=True)
ylabs = list(a.index)
xlabs = list(a.columns)

fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(a, cmap='RdBu_r', vmin=0, vmax=max_val,
               extent=[0, len(xlabs), 0, len(ylabs)],
               interpolation='none', origin='lower')

# We want to show all ticks...
plt.xticks([x - 0.5 for x in list(range(1, len(xlabs) + 1))], xlabs, rotation=90)
plt.yticks([x - 0.5 for x in list(range(1, len(ylabs) + 1))], ylabs)

# ... and label them with the respective list entries
ax.set_xticklabels(xlabs)
ax.set_yticklabels(ylabs)
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Floods disrupting the preferred shortest route', rotation=90)
plt.title('Amount of floods (AoI\'s) per preferred shortest route')
plt.show()
fig.savefig(os.path.join(save_figs, 'heatmap_dist_floods_per_route.png'))
plt.close()


# ## OD PAIR ANALYSIS RESULTS ##
# # plot the aggregated results from the OD pairs analysis
# df = gpd.read_file(r"N:\Projects\11202000\11202067\F. Other information\Flooding and transport disruptions\2_Output_analyses_Network\4_multi_link_one_aoi_time\AoI_rp100_OD_pairs_time.shp")
#
# # AoI's per preferred route
# ax = df.groupby('AoI').size().sort_values(ascending=False).plot.hist(bins=10)
# plt.title("Number of routes affected by a single flood event (AoI)")
# plt.xlabel("Disrupted preferred routes")
# plt.ylabel("count")
# plt.show()
# plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_disrupted_detour_lines.png")
# plt.close()
#
# # preferred routes per AoI
# ax = df.groupby(['origin', 'destinatio']).size().sort_values(ascending=False).plot.hist(bins=5)
# plt.title("Number of flood events (AoI's) affecting a preferred route")
# plt.xlabel("Number of flood events (AoI's)")
# plt.ylabel("count")
# plt.show()
# plt.savefig(r"P:\osm_flood\network_analysis\stochastic_results\figures\austria_disrupted_detour_lines.png")
# plt.close()


# check how the preferred routes are different per distance/time
total = len(pref_dist.index)
similar = 0
for i in range(len(pref_dist.index)):
    orig, dest = pref_dist.iloc[i, 2:4]
    geom_time = pref_time.loc[(pref_time['origin'] == orig) & (pref_time['destinatio'] == dest), 'geometry'].iloc[0]
    if geom_time.equals(pref_dist['geometry'].iloc[i]):
        similar += 1

perc_similar = similar / total * 100


# Visualize the density of the OD routes
# filter first the dataset
# pt = pref_time.loc[pref_time['odpair'].isin(list(df.loc[df['disrupted'] == '1', 'odpair']))]
pt = gpd.read_file(r"D:\COACCH_countries\sweden\time_pref_routes.shp")

gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs={'init': 'epsg:4326'})
row = 0
# create a dataframe with all linestrings
for i in range(len(pt)):
    print("{}%".format(round(i / len(pt) * 100, 0)))
    for line in pt.geometry.iloc[i]:
        gdf.loc[row] = [line]
        row += 1

geom_hash = [hash(tuple(geom.coords)) for geom in gdf.geometry]
gdf['geom_hash'] = geom_hash

result = gdf.groupby('geom_hash').size().reset_index()
result = gpd.GeoDataFrame(result.merge(gdf, how='left', on='geom_hash'), geometry='geometry', crs={'init': 'epsg:4326'})
result.drop_duplicates('geom_hash', inplace=True)
result.rename(columns={0:'count'}, inplace=True)

result.to_file(r"D:\COACCH_countries\sweden\pref_time_density.shp",
               encoding='utf-8')
