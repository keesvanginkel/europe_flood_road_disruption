# -*- coding: utf-8 -*-
"""
Created on 2-12-2020

@author: Frederique de Groen

Part of a COACCH criticality analysis of networks.

"""

import pygeos
import pandas as pd

networks_europe_path = r'D:\COACCH_paper\data\networks_europe_elco_koks'



network = pd.read_feather(‘LUX-edges.feather’)
network.geometry = pygeos.from_wkb(network.geometry)
