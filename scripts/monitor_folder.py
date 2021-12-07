from pathlib import Path
import os
import tqdm as tqdm
import time as time


to_eval = Path("D:\Europe_percolation\combined_results\combined_main_output")
assert to_eval.exists()

countries = list(to_eval.glob('*'))
print(to_eval)
countries_dict = {} #keys = country name, value = paths
for c in countries:
    countries_dict[c.stem] = c

input_txt = 'nothing'
while input_txt not in list(countries_dict.keys()):
    input_txt = input('Enter a country for which you want to monitor the progress, '
                      'you may choose from: {}'.format(list(countries_dict.keys())))
    if input_txt not in list(countries_dict.keys()):
       print('Please enter a valid country name!')

print('We will try to make a progress bar for {}, by evaluating folder: {}'.format(input_txt,countries_dict[input_txt]))

to_eval = countries_dict[input_txt]
scheduled_p = to_eval / 'scheduled'
finished_p = to_eval / 'finished'

if not scheduled_p.exists():
    raise OSError(2,
                  'Directory with scheduled percolation analysis does not exists, maybe the percolation scheduler is not yet run?',
                  '{}'.format(scheduled_p))

if not finished_p.exists():
    raise OSError(2,
                  'Directory with percolation results does not exist, maybe the percolation analysis is not started yet?',
                  '{}'.format(finished_p))

start = time.time()
print('Timer started at: ',time.asctime( time.localtime(start) ))

# Most simple approach: compare number of files in scheduled_p
n_scheduled = len(list(scheduled_p.glob('**/*.pkl')))  # number of pickles scheduled
initial_finished = len(list(finished_p.glob('**/*.csv')))
n_finished = initial_finished
old_value = initial_finished
with tqdm.tqdm(total=n_scheduled,initial=initial_finished) as pbar:
    while n_finished < n_scheduled:
        time.sleep(10)
        n_finished = len(list(finished_p.glob('**/*.csv')))
        increment = n_finished - old_value
        pbar.update(increment)
        old_value = n_finished
pbar.close()
print('We think {} is finished.'.format(input_txt))
end = time.time()
print('Timer ended at: ',time.asctime( time.localtime(start) ))
print('Etimated runtime: {:.1f} seconds,'.format(end-start),
      'i.e. {:.1f} minutes'.format((end-start)/60),
      'i.e. {:.1f} hours'.format((end-start)/3600),
      'hours, i.e {:.1f} days'.format((end-start)/(3600*24)))
