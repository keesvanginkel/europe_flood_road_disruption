"""
Script for static evaluating the progress of multiple countries.
For dynamic evaluating progress over time (progress bar) see monitor_folder.py

"""


from pathlib import Path
import os
import tqdm as tqdm
import time as time
import pandas as pd

to_eval = Path(r"P:\osm_flood\network_analysis\data\main_output")  # Path to evaluate
#to_eval = Path(r"P:\osm_flood\network_analysis\igraph")
assert to_eval.exists()

countries = list(to_eval.glob('*'))
print(to_eval)
countries_dict = {} #keys = country name, value = paths
for c in countries:
    countries_dict[c.stem] = c

countries_list = list(countries_dict.keys())
df = pd.DataFrame(index=countries_list,columns=['scheduled','finished','percentage_finished'])

print('Counting files in folders:')
for c in tqdm.tqdm(countries_list):
    scheduled_p = countries_dict[c] / 'scheduled'
    finished_p = countries_dict[c] / 'finished'
    if not scheduled_p.exists():
        sched = 0
    else:
        sched = len(list(scheduled_p.glob('**/*.pkl')))  # number of pickles scheduled
    if not finished_p.exists():
        fin = 0
    else:
        fin = len(list(finished_p.glob('**/*.csv')))

    if sched == 0:
        perc = None
    else:
        perc = 100 * fin/sched

    df.at[c,'scheduled'] = sched
    df.at[c,'finished'] = fin
    df.at[c,'percentage_finished'] = perc

print(df)

