# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:23:40 2023

Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script filters ASOS/AWOS data downloaded from the Iowa Environmental
Mesonet https://mesonet.agron.iastate.edu/request/download.phtml?network=FL_ASOS.
Since the downloads are per date, they are comprised of every station in Florida's
5 minute observation.  This script will output the 24-hour summation of precipitation
per station ID (based on times within the sheet ex: midnight to midnight, or 8am to 8am).
"""

import pandas as pd

# Identify file
asos_file = r'X:/MGIS/Capstone/data/asos/asos_20230831_fl12Zto12Z.csv'

# Read in the file, create unique series for lat, lon, and elevation, and sum
# the precip over the 24 hour period
df = pd.read_csv(asos_file, index_col = 'valid')
df = df.set_index(pd.to_datetime(df.index))

ser_list = [] # Create empty list to store values by station

# Loop through the stations in the dataframe
for station in df.station.unique():
    # Print station to track progress
    print (station)

    # Filter the dataframe by the station
    ser = df[df.station==station]

    # Create dataframe of station and the hourly maximum precipitation value.
    # This works with because ASOS will display the accumulated precipitation
    # during each time step of the hour, then reset once the hourly ob is submitted
    # which usually occurs around xx:53 but not always. Taking the max each hour gracefully
    # handles this potential hour. The sum of the hourly maximum values of precipitation
    # will be the daily preciptation.
    ser_list.append(pd.DataFrame([station,ser.resample('H')['p01i'].max().sum()]))

# The dataframes are concatenated along axis 1, then transposed to turn into columns.
# The columns are renamed and station is set as the index for similarity with other dataframes
# below (lat, lon, elev). Finally, the dataframe is rounded off at 2 decimals.
precip_series = pd.concat(ser_list,axis=1)
precip_series = precip_series.transpose()
precip_series.columns = ['station','p01i']
precip_series.set_index('station',inplace=True)
precip_series = precip_series.astype(float).round(2)

# Lat, lon, elev sereis are more straightforward. 
lat_series = df.groupby(['station'])['lat'].first()
lon_series = df.groupby(['station'])['lon'].first()
elev_series = df.groupby(['station'])['elevation'].first()

# Create new dataframe and rename columns
new_df = pd.concat([precip_series, lat_series, lon_series, elev_series], axis=1)
new_df.columns = ['24hr_precip_in', 'lat', 'lon', 'elev_m']

# Export to CSV
new_df.to_csv(r'X:/MGIS/Capstone/data/asos/24hr_12zto12z_asos_20230831.csv')

# End program