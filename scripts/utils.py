import os
import json
from pathlib import Path


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

if __name__ == '__main__':
    print('utils.py is running a test procedure')
    print('Existence of config paths is tested')
    config = load_config()
    for key, path in config['paths'].items():
        print(key, path, path.exists())