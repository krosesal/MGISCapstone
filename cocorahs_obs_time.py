
"""
Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script scrubs the COCORAHS Data Explorer in order to retrieve the observation times for the data presented
in the ACIS dataset. https://dex.cocorahs.org/ 
"""

import requests
import pandas as pd
import numpy as np

# First, read in CSV with all ACIS data in order to parse out each individual network
# COCORAHS = 10 https://www.rcc-acis.org/docs_webservices.html
date = '2023-09-01'
network = 'COCORAHS'
ident = '10'
acis_file = r'X:/MGIS/Capstone/data/acis/' + date + '_acis.csv'
acis_df = pd.read_csv(acis_file, header=0)

network_df = acis_df[acis_df[network.lower()]==True]

# Obtain the stations ids for only the network of interest
sids = []

for index,row in network_df.iterrows():
    staType_df = pd.DataFrame()
    staType_df['sta_type'] = row['sta_type'].replace('\'','').strip('[]').split(', ')
    staType_df['sids'] = row['sids'].replace('\'','').strip('[]').split(', ')
    sid = staType_df[staType_df['sta_type']==ident]['sids'].values[0]
    sid_fmt = 'FL-' + sid[2:4] + '-' + str(int(sid[4:8]))
    sids.append(sid_fmt)

network_df[network + '_sids']=sids

# Remove records that are lacking precipiation data
network_df = network_df.dropna(axis=0)

# Initilize a list for the times, and use the station ids to grab the obs times from the COCORAHS data explorer.
# Remove any stations that either have no precip observation or daily observation time
time_list = []

for station in network_df[network + '_sids']:
    md = pd.read_html('http://dex.cocorahs.org/stations/' + station + '/obs-tables?from=' + date + '&to=' + date)
    obsTime = md[1][2][0]
    print (station, obsTime)
    time_list.append(obsTime)

# Append the observation times back into the dataframe, then export a CSV with only the necessary information
network_df['obsTime'] = time_list
network_df[['Unnamed: 0', 'lon', 'lat', 'elev_ft', network + '_sids', 'precip_inches', 'obsTime']].to_csv(r'X:/MGIS/Capstone/data/acis/' + network + '_' + date + '.csv')
print ('cheese knife')

# End Program