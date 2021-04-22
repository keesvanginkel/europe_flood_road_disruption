"""
This file creates utils functions which are specific for processing data on the European scale

Author: Kees van Ginkel
"""

import pandas as pd
import warnings

from utils import load_config

# Load configuration file
config = load_config()

def N0_to_3L(NUTS_codes):
    """From NUTS to 3L code
    Accepts string or list of strings e.g, 'DE' ['NL,'BE']
    Returns in the same format

    df contains data: replace with import from config
    """
    data = config['paths']['data']
    df = pd.read_csv((data / 'country_codes.csv'), delimiter=';', index_col='code2')

    unpack = False
    if not isinstance(NUTS_codes, list):
        NUTS_codes = [NUTS_codes]
        unpack = True
    sel = list(df.loc[df.index.isin(NUTS_codes)]['code3'].values)
    if unpack: sel = sel[0]
    return sel


def L3_to_N0(L3_codes):
    """From 3L country code to NUTS0-code
    Accepts string or list of strings e.g, 'DEU' ['BEL,'NED']
    Returns in the same format

    df contains data: replace with import from config
    """
    data = config['paths']['data']
    df = pd.read_csv((data / 'country_codes.csv'), delimiter=';', index_col='code3')

    unpack = False
    if not isinstance(L3_codes, list):
        L3_codes = [L3_codes]
        unpack = True
    sel = list(df.loc[df.index.isin(L3_codes)]['code2'].values)
    if unpack: sel = sel[0]
    return sel

def country_names(country_codes):
    """Find country names for given 2 letter or 3 letter codes
    Accepts string or list of strings e.g, 'DEU' ['BEL,'NED'],
    2l and 3l codes may be mixed e.g. ['NED', 'BE']

    df contains data: replace with import from config
    """
    data = config['paths']['data']
    df = pd.read_csv((data / 'country_codes.csv'), delimiter=';', index_col='code3')
    df_3l = df['country']
    df_2l = df.set_index('code2').squeeze(axis=1)
    df_3l.index.name = 'code'
    df_2l.index.name = 'code'
    df = df_2l.append(df_3l)

    unpack = False
    if not isinstance(country_codes, list):
        country_codes = [country_codes]
        unpack = True
    sel = list(df.loc[df.index.isin(country_codes)].values)
    if unpack: sel = sel[0]
    return sel


def NUTS_3_remote(**kwargs):
    """
    Returns a list with remote NUTS-3 regions you probably don't want to plot

    Optional keyword arguments (boolean):
            "Overseas"    : Removes remote, overseas areas (default True)
            "Creta"       : Greek island Creta (default False)
            "Spain"       : Ceauto and Melilla (Spanish North Coast) (default True)

    #Suggested syntax for filtering a list:
    [e for e in LIST if e not in NUTS_3_remote()]

    From OSdaMage
    """
    options = {'Overseas': True, 'Creta': False, 'Spain': True}  # default keyword arguments
    options.update(kwargs)

    l = []

    if options['Overseas']:  # Remote, overseas areas
        l.extend(['PT200', 'PT300',  # Portugal: Azores and Madeira
                  'ES703', 'ES704', 'ES705', 'ES706', 'ES707', 'ES708', 'ES709',  # Spain: Canary Islands
                  'FRY10', 'FRY20', 'FRY30', 'FRY40',
                  'FRY50'])  # France: overseas areas: Gouadeloupe, Martinique, French Guiana, La Reunion, Mayotte "])

    if options['Creta']:
        l.extend(['EL431', 'EL432', 'EL433', 'EL444'])

    if options['Spain']:  # Ceauto and Melilla: autonomous Spanish city at the North Coast of Africa
        l.extend(['ES630', 'ES640'])

    l.sort()

    return l

if __name__ == '__main__':
    #print(config)

    #Run a simple test procedure
    data = config['paths']['data']
    if not data.exists():
        warnings.warn('The output data folder {}, as set in the config, does not exist.'.format(output_data))

    if not (data / 'country_codes.csv').exists():
        warnings.warn('Output data folder misses file country_codes.csv, NUTS-letter conversion function will not work.'.format(output_data))

    print(N0_to_3L(['BE','NL']))
    print('ho es')