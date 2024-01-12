# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:16:15 2023

Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script grabs archived data from Applied Climate Information System
(ACIS).  This data includes observations from CoCoRAHS, GHCN, COOP, Threadex, and
WBAN, and includes fields of precipitation type, amount, temperature, and snow.
Metadata accessed here: https://www.rcc-acis.org/docs_webservices.html.
Code advised by Anthony Preucil.
"""

import requests
import pandas as pd
import numpy as np
import re

startDate = '20230829'
endDate = '20230831'

print ('Reading Data From Online...')
res = requests.get('http://data.rcc-acis.org/MultiStnData?bbox=-86.1,26.0,-81.3,30.6&sdate='+startDate+'&edate='+endDate+'&elems=4&meta=name,ll,elev,sids')
data_json = res.json()
print ('Done!')

names = [d['meta']['name'] for d in data_json['data']]

print ('Saving Metadata for later...')
# Create a dataframe with metadata for each station (latitude and longitude)
meta = pd.DataFrame(index = names)
meta['lon'] = [d['meta']['ll'][0] for d in data_json['data']]
meta['lat'] = [d['meta']['ll'][1] for d in data_json['data']]
elevation = []
for m in data_json['data']:
    try:
        elevation.append(m['meta']['elev'])
    except:
        elevation.append(np.NaN)      
meta['elev_ft'] = elevation
meta['sids'] = [d['meta']['sids'] for d in data_json['data']]
station = []
for s in meta['sids']:
    station.append([re.split('\s', sid)[1] for sid in s])    
meta['sta_type'] = station
meta['wban'] = [True if '1' in st else False for st in meta['sta_type']]
meta['coop'] = [True if '2' in st else False for st in meta['sta_type']]
meta['faa'] = [True if '3' in st else False for st in meta['sta_type']]
meta['wmo'] = [True if '4' in st else False for st in meta['sta_type']]
meta['icao'] = [True if '5' in st else False for st in meta['sta_type']]
meta['ghcn'] = [True if '6' in st else False for st in meta['sta_type']]
meta['nwsli'] = [True if '7' in st else False for st in meta['sta_type']]
meta['threadex'] = [True if '9' in st else False for st in meta['sta_type']]
meta['cocorahs'] = [True if '10' in st else False for st in meta['sta_type']]

print ('Done!')

# Create a list of series for each station's data
print ('Combining Data from each station into one dataframe...')
series_list = []
for n, d in zip(names, data_json['data']):
    series = pd.Series([p[0] for p in d['data']], name=n)
    series_list.append(series)
   
# Concatenate the list of series to create the dataframe
precip_grid_stations = pd.concat(series_list, axis=1)
print ('Done!')
print ('Cleaning Up the data...')
# Replace M (missing value) with NaN and T (trace value) with 0.
precip_grid_stations = precip_grid_stations.replace('M', np.nan)
precip_grid_stations = precip_grid_stations.replace('T', 0)
   
# Convert the datatype to numeric for all daily precipitation values in the dataframe
precip_grid_stations = precip_grid_stations.apply(pd.to_numeric, errors='coerce')
   
# Create an index column of daily dates
dates = pd.date_range(startDate, endDate)
precip_grid_stations['Date'] = dates
precip_grid_stations = precip_grid_stations.set_index('Date')
print ('Done!')

# Create exportable CSV per date
precip_grid_stations = precip_grid_stations.transpose()
for d in precip_grid_stations.columns:
    df = pd.concat([meta, precip_grid_stations[d]], axis=1)
    df = df.rename(columns={d: 'precip_inches'})
    df.to_csv(r'X:/MGIS/Capstone/data/acis/' + str(d)[:10] + '_acis.csv')
    

    
# End program