"""
This script will access data from the Weather Underground network via API key. 
"""


import json
import requests
import pandas as pd

# Create CSV with list of desired stations
st_file = r'X:/MGIS/Capstone/data/wunderground/stations.csv'
df = pd.read_csv(st_file, header=0)

# Set up parameters for access URL.  Additional documentation: https://docs.google.com/document/d/1eKCnKXI9xnoMGRRzOL1xPCBihNV2rOet08qpE_gArAY/edit
api_key = ''  # add your api key here
#product = r'v2/pws/history/daily'  #https://docs.google.com/document/d/1w8jbqfAk0tfZS5P7hYnar1JiitM0gQZB-clxDfG3aD0/edit 
product = r'v2/pws/history/all'
station_id = df['st_name']
print(station_id)
exp_form = 'json'  #or xml
date = '20230830'

df_final = pd.DataFrame()
df_list = [] # hello

for sid in station_id:
    f = requests.get(r'https://api.weather.com/' + product + r'?stationId=' + sid + r'&format=' + exp_form + r'&units=e&date=' + date + r'&apiKey=' + api_key)
    json_string = f.json()
    #print(json_string['observations'][0])

# Create dataframes from returned json so that they can be made into shapefiles
    try:
        data_dict = json_string['observations'][0].pop('imperial')
    except:
        print(sid)
        continue
    meta_dict = json_string['observations'][0]
    meta_dict.update(data_dict)
    df = pd.DataFrame.from_dict(meta_dict, orient='index', columns=[sid])
    df = df.transpose()
    df_list.append(df)
    f.close()

df_final = pd.concat(df_list)


# Export to CSV
df_final.to_csv(r'X:/MGIS/Capstone/data/wunderground/daily_wunderground_' + date + r'.csv')

# End program