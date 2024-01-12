# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:23:40 2023

Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script creates a 24 hour file for the Wunderground data, based on the hourly precipiation readings
from 12z to 12z.  Example: 20230830 = 12z 0829 to 12z 0830.
"""

import pandas as pd

# Identify file
wx_file = r'X:/MGIS/Capstone/data/wunderground/hourly_wunderground_12Zto12Z_20230829.csv'

# Read in the file, create unique series for lat, lon, and elevation, and sum
# the precip over the 24 hour period
df = pd.read_csv(wx_file, index_col = 'obsTimeLocal')
df = df.set_index(pd.to_datetime(df.index))

ser_list = [] # Create empty list to store values by station

# Loop through the stations in the dataframe
for station in df.stationID.unique():
    # Print station to track progress
    print (station)

    # Filter the dataframe by the station
    ser = df[df.stationID==station]

    # Sum the total precipitation for the 24hr period
    ser_list.append(pd.DataFrame([station,ser['precipHourly'].sum()]))

# The dataframes are concatenated along axis 1, then transposed to turn into columns.
# The columns are renamed and station is set as the index for similarity with other dataframes
# below (lat, lon, elev). Finally, the dataframe is rounded off at 2 decimals.
precip_series = pd.concat(ser_list,axis=1)
precip_series = precip_series.transpose()
precip_series.columns = ['stationID','precipHourly']
precip_series.set_index('stationID',inplace=True)
precip_series = precip_series.astype(float).round(2)

# Lat, lon sereis are more straightforward. 
lat_series = df.groupby(['stationID'])['lat'].first()
lon_series = df.groupby(['stationID'])['lon'].first()

# Create new dataframe and rename columns
new_df = pd.concat([precip_series, lat_series, lon_series], axis=1)
new_df.columns = ['24hr_precip_in', 'lat', 'lon']

# Export to CSV
new_df.to_csv(r'X:/MGIS/Capstone/data/wunderground/24hr_12zto12z_wunderground_20230829.csv')

# End program