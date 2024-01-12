# -*- coding: utf-8 -*-
"""
MGIS Capstone Project

Author:
Kristina Preucil
krs5664@psu.edu

5/16/2023

This script accesses the mPING database, and returns reports in JSON format.
This script was written based on the examples provided by Oklahoma University
at this website: https://mping.ou.edu/api/api/examples.html.

For more filter (data and spatial) requests, see here: https://mping.ou.edu/api/api/reports/filters.html

Eventually, this script will be implemented into an ArcPro Script Tool so that
data requests can be streamlined, and plotted into GIS.

My mPING API token was activated 4/24/2023 by Jeff Brogden (jeff.brogden@noaa.gov).
Note, this API is active under a research license allowance, and cannot be used
to garner any type of income.

User: krosesal
Token key: 15e1e26d6b6d75e843702d568886d40f95f1ee7d


JSON variables: id, obtime, category, description, description_id, geom

mPing categories: Test, None, Rain/Snow, Hail, Wind Damage, Tornado, Flood, Mudslide, Reduced Visibility, Winter Weather Impacts

Rain/Snow descriptions:  Rain, Freezing Rain, Drizzle, Ice Pellets/Sleet, Snow and/or Graupel, Mixed Rain and Snow,
   Mixed Ice Pellets and Snow, Mixed Freezing Rain and Ice Pellets, Mixed Rain and Ice Pellets

Hail descriptions: Pea (0.25 in.), Half-inch (0.50 in.), Dime (0.75 in.), Quarter (1.00 in.), Half Dollar (1.25 in.),
   Ping Pong Ball (1.50 in.), Golf Ball (1.75 in.), Hen Egg (2.00 in.), Hen Egg+ (2.25 in.), Tennis Ball (2.50 in.),
   Baseball (2.75 in.), Tea Cup (3.00 in.), Baseball+ (3.25 in.), Baseball++ (3.50 in.), Grapefruit- (3.75 in.),
   Grapefruit (4.00 in.), Grapefruit+ (4.25 in.), Softball (4.50 in.), Softball+ (4.75 in.), Softball++ (>=5.00 in.)

Wind Damage descriptions: Lawn furniture or trash cans displaced; Small twigs broken, 1-inch tree limbs broken; Shingles blown off,
   3-inch tree limbs broken; Power poles broken, Trees uprooted or snapped; Roof blown off, Homes/Buildings completely destroyed

Tornado descriptions: Tornado (on ground), Water Spout

Flood descriptions: River/Creek overflowing; Cropland/Yard/Basement Flooding, Street/road flooding; Street/road closed; Vehicles stranded,
   Homes or buildings filled with water, Homes, buildings or vehicles swept away

Reduced Visibility descriptions: Dense Fog, Blowing Dust/Sand, Blowing Snow, Snow Squall, Smoke

Winter Weather Impacts descriptions: Downed tree limbs or power lines from snow or ice, Frozen or burst water pipes,
   Roof or structural collapse from snow or ice, School or business delay or early dismissal, School or business closure,
   Power or internet outage or disruption, Road closure, Icy sidewalks, driveways, and/or parking lots,
   Snow accumulating only on grass, Snow accumulating on roads and sidewalks
"""

# Import needed modules
import requests
import json
import pandas as pd
import arcpy

# Set up variables
output_folder = "X:\MGIS\Capstone\code_tests"
output_file = 'test_idalia_20230830.shp'
full_file = output_folder + r'//' + output_file
sr = arcpy.SpatialReference(4326)

# Set up our request headers indicating we want json returned and include
# our API Key for authorization.
# Make sure to include the word 'Token'. ie 'Token yourreallylongapikeyhere'
reqheaders = {
    'content-type':'application/json',
    'Authorization': 'Token 15e1e26d6b6d75e843702d568886d40f95f1ee7d'
    }

# Define the parameters here which you want to filter the data by
reqparams = {
    'category':'Flood',
    #'description':'Pea (0.25 in.)',
    'year':'2023',
    'month':'08',
    'day':'30'
}

url = 'http://mping.ou.edu/mping/api/v2/reports'
response = requests.get(url, params=reqparams, headers=reqheaders)

#print(response.url)

# Let the user know if request worked or not, then print the returned JSON dictionaries
if response.status_code != 200:
    print('Request Failed with status code ' + str(response.status_code))
else:
    print('Request Successful')
    data = response.json()
    # Pretty print the data
    #print(json.dumps(data,indent=4))

# Access only the point coordinates in returned dictionaries and put them into
# their own list
results_df = pd.DataFrame(data['results'])
geom_df = pd.DataFrame(results_df['geom'])
coords = []

for index,i in geom_df.iterrows():
    coords.append(i['geom']['coordinates'])

# Append coordinates into the results data frame
results_df['Coordinates'] = coords


# Create shapefile and fields
arcpy.management.CreateFeatureclass(output_folder, output_file, geometry_type="POINT", spatial_reference=sr)
arcpy.management.AddField(full_file, 'ping_id', "LONG")
arcpy.management.AddField(full_file, 'date_time', "STRING", field_length=30)
arcpy.management.AddField(full_file, 'category', "STRING", field_length=30)
arcpy.management.AddField(full_file, 'descript', "STRING", field_length=60)
arcpy.management.AddField(full_file, 'lon', "FLOAT")
arcpy.management.AddField(full_file, 'lat', "FLOAT")

fields = ['SHAPE@XY', 'ping_id', 'date_time', 'category', 'descript', 'lon', 'lat']

with arcpy.da.InsertCursor(full_file, fields) as cursor:
    for index,x in results_df.iterrows():
        cursor.insertRow((x['Coordinates'], x['id'], x['obtime'], x['category'], x['description'], x['Coordinates'][0], x['Coordinates'][1]))

del cursor


# End program


