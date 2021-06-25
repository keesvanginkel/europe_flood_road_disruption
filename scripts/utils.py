import os
import json
from pathlib import Path
from shutil import copyfile



def load_config():
    """
    Read config.json
    NOTE: make sure your working directory is set to the highest level folder in the directory

    Arguments:
        No arguments

    Returns:
        *config* (dict) : Dict with structure:
            ['comment'] : 'String with info about the config"
            ['paths'] [key] : Pathlib path to folders
                      [another key]
    @author: Kees van Ginkel
    """
    config_path = Path(__file__).parents[1] / 'config.json'

    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)

    #Convert to pathlib objects
    for key in ['paths']:
        for key2, value in config[key].items():
            config[key][key2] = Path(__file__).parents[1] / Path(value)
    return config

def highway_mapper():
    "Returns dict with mapping of OSM highway keys to superclasses"
    highway_mapper = {
        'motorway_link' : 'motorway',
        'motorway' : 'motorway',
        'trunk' : 'trunk',
        'trunk_link' : 'trunk',
        'primary' : 'primary',
        'primary_link' : 'primary',
        'tertiary' : 'tertiary',
        'tertiary_link' : 'tertiary',
        'secondary' : 'secondary',
        'secondary_link' : 'secondary',
        'tertiary' : 'tertiary',
        'tertiary_link' : 'tertiary'}
    return highway_mapper

def smart_AoI_copy(origin,destination,skip=[None]):
    """
    Copies the aggregated (step 1) AoI results
    Rather than copying all individual experiments, this function copies the results per number of combinations (one level higher)
    
    Expects origin with these subfolders:
        - {country}
            - finished
              - .csv files (copies these files)
             (-) probably also has directories with the individual results, these are not copied
    
    Arguments:
        *origin* (Path) : origin folder
        *destination* (Path) : destination folder
        
    """
    assert (origin.exists() and destination.exists())

    origin_country_paths = [x for x in origin.iterdir() if x.is_dir()]
    for p in origin_country_paths:
        if p.stem in skip:
            print('Not copying {}'.format(p.stem))
            continue
        else:
            destdir = destination / p.stem / 'finished' #make subfolder country / finished in dest dir
            destdir.mkdir(parents=True)
            finished = p / 'finished' #folder where the aggregated AoI results should be located
            csv_files = [x for x in finished.iterdir() if x.suffix == '.csv']
            csv_aois = [int(x.stem.split('_')[1]) for x in csv_files]
            print(p.stem,'has files for:',sorted(csv_aois) ,'combinations of AoI')
            for sourcefile in csv_files:
                destfile = destdir / sourcefile.name  
                copyfile(sourcefile,destfile)
            print('Copying for {} finished'.format(p.stem))

### Example use:
# origin = Path("P:\\osm_flood\\network_analysis\\igraph")
# destination = Path("D:\\Europe_percolation\\combined_results\\combined_main_output")
# skip = ['.idea','europe_flood_road_disruption','trails','figures']
# skip.extend(['albania','austria','belgium','bulgaria','croatia','czechia','denmark'])
# print(skip)
# smart_AoI_copy(origin,destination,skip=skip)

if __name__ == '__main__':
    print('utils.py is running a test procedure')
    print('Existence of config paths is tested')
    config = load_config()
    for key, path in config['paths'].items():
        print(key, path, path.exists())

