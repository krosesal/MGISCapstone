
"""
Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script accesses Historical Observing Metadata Repository (HOMR) in order to retrieve observation times
for stations associated with the ACIS datasets. We will focus on COOP.
https://www.ncei.noaa.gov/access/homr/api;jsessionid=64EF7DC85249103A4E357808B715F800
"""

import requests
import pandas as pd
import numpy as np

# First, read in CSV with all ACIS data in order to parse out each individual network
# COOP = 2 https://www.rcc-acis.org/docs_webservices.html
date = '2023-08-28'
network = 'COOP'
ident = '2'
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
    sids.append(sid[:-2])

network_df[network + '_sids']=sids

# Remove records that are lacking precipiation data
network_df = network_df.dropna(axis=0)

# Initilize a list for the times, and use the station ids to grab the obs times from HOMR.
# Remove any stations that either have no precip observation or daily observation time
time_list = []

for station in network_df[network + '_sids']:
    print ('Reading Data From Online...', network, station)
    res = requests.get('http://www.ncdc.noaa.gov/homr/services/station/search?qid=' + network + ':' + station + '&date=' + date)
    data_json = res.json()
    print ('Done!')

    try:
        elem_dict = data_json['stationCollection']['stations'][0]['elements']
    except:
        print (network + ': ' + station + ' dun broke')
        network_df = network_df.drop(network_df[network_df[network + '_sids']==station].index)
        continue

    elem_list = [d['element'] for d in elem_dict]
    if 'PRECIP' in elem_list:
        pass
    else:
        print ('Removing', station, 'due to no PRECIP obs time')
        network_df = network_df.drop(network_df[network_df[network + '_sids']==station].index)

    c = 0
    for d in elem_dict:
        if d['element'] == 'PRECIP' and d['frequency'] == 'DAILY':
            obsTime = d['observationTime']
            print (obsTime)
            time_list.append(obsTime)
            c+=1
            if c > 1:
                print (station + ' has multiple obs times for precip')
        else:
            pass

# Append the observation times back into the dataframe, then export a CSV with only the necessary information
network_df['obsTime'] = time_list
network_df[['Unnamed: 0', 'lon', 'lat', 'elev_ft', network + '_sids', 'precip_inches', 'obsTime']].to_csv(r'X:/MGIS/Capstone/data/acis/' + network + '_' + date + '.csv')
print ('chickens')

# End Program