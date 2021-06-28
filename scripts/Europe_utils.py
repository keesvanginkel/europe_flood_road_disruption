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


def country_code_from_name(country_names,l3=False):
    """2 letter ['BE'] or 3 letter codes ['BEL'] from country names
    Accepts string or list of strings e.g, 'Serbia' or ['Belgium','Slovakia']
    
    Arguments:
        *country_names* (string or list of strings) : country names
        *l3* (Boolean) : return 3l-code; default = False -> returns 2l-code
        
    Returns
        *sel* (string or list of strings) : 2l or 3l codes
    """

    if True:
        data = config['paths']['data']
        df = pd.read_csv((data / 'country_codes.csv'), delimiter=';')
        df_3l = df['country']

        if l3:
            code_col = 'code3' #return 3l code
        else:
            code_col = 'code2' #return 2l code

        unpack = False
        if not isinstance(country_names, list):
            country_names = [country_names]
            unpack = True
        sel = list(df.loc[df.country.isin(country_names)][code_col])
        if unpack: sel = sel[0]
    return sel

def ignore_countries():
    """
    Returns a list of countries (NUTS-0 codes) that are not included in the analysis
    *countries* (list) : countries not included in analysis
    """
    countries = ['CY','IS','LI','LU','ME','MT'] #too little NUTS-regions (2 or less on NUTS-3)
    countries.append('TR') #outside flood hazard domain
    return countries
    

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

def NUTS_3_islands():
    """Filter out additional Islands, which are not included in NUTS_3_remote()
    
    Returns a list with NUTS-3 regions you probably do not want to include in a road network analysis

    Todo: sync with preprocess_routes.py line 42-63 (overseas = [])
    """
    islands = ['FRM01','FRM02'] #Corse
    islands.extend(['UKN06','UKN07','UKN08','UKN09','UKN10','UKN11','UKN12',
                    'UKN13','UKN14','UKN15','UKN16']) #Northern Island, not connected to mainland UK
    islands.extend(['ITG25','ITG26','ITG27','ITG28','ITG29','ITG2A','ITG2B','ITG2C']) #IT Sardegna
    islands.extend(['ITG11','ITG12','ITG13','ITG14','ITG15','ITG16','ITG17','ITG18','ITG19']) #IT Sicilia
    return islands

def NUTS_2_islands():
    """List of islands on NUTS-2 level, which are not included in NUTS_2_remote()
    """
    islands = ["FRM0"] #Corse
    islands.extend(['UKN0']) #Northern island
    islands.extend(['ITG1','ITG2']) #Sicilia and Sardegna
    return islands

def create_gridlines(ps,ms,point_spacing=1000):
    """
    Create GeoSeries containing parallels in WGS84 projection.
    
    Arguments:
        *ps* (list) - Parallel coordinates (degrees) of the lines to plot (e.g. [40,50,60,70])
        *ms* (list) - Meridian coordinates (degrees) of the lines to plot (e.g. [-30,-20,-10,0,10,20,30,40,50,60,70])
        *pointspacing* (integer) - Number of points to create (to draw smooth line) (e.g. 100)
    
    Returns:
        *P_series,M_series* (Geopandas GeoSeries) - Contains the parallels and meridians
    """
    import numpy as np
    import geopandas as gpd 
    from shapely.geometry import LineString
    
    #create parallels
    Parallels = []
    start = ms[0]
    end = ms[-1]
    x_values = np.linspace(start,end,point_spacing)

    for p in ps:
        Points = []
        for x in x_values:
            point = (x,p)
            Points.append(point)
        Parallel = LineString(Points)
        Parallels.append(Parallel)

    P_series = gpd.GeoSeries(Parallels,crs='EPSG:4326')
    

    #create meridians
    Meridians = []
    start = ps[0]
    end = ps[-1]
    y_values = np.linspace(start,end,point_spacing)

    for m in ms:
        Points = []
        for y in y_values:
            point = (m,y)
            Points.append(point)
        Meridian = LineString(Points)
        Meridians.append(Meridian)

    M_series = gpd.GeoSeries(Meridians,crs='EPSG:4326')
    
    return P_series,M_series


if __name__ == '__main__':
    #print(config)

    #Run a simple test procedure
    data = config['paths']['data']
    if not data.exists():
        warnings.warn('The output data folder {}, as set in the config, does not exist.'.format(output_data))

    if not (data / 'country_codes.csv').exists():
        warnings.warn('Output data folder misses file country_codes.csv, NUTS-letter conversion function will not work.'.format(output_data))

    print(N0_to_3L(['BE','NL']))

