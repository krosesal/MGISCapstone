"""
MGIS Capstone Project

Author:
Kristina Preucil
krs5664@psu.edu

3/18/2024

This script attempts to augment the MRMS daily precipitation output by adding influence from
Weather Underground Personal Weather Stations (WU) and COCORAHS and NWS COOP rain gauges.  The
new MRMS model is being compared to the PRISM model.  It first resamples all of the inputs so
that they are the same size in order for the weighting masks to function.

The second portion of the code creates the difference plots and the scatter plots for the initial
MRMS - PRISM and the New MRMS - PRISM.  In order to experiment with these outputs, the total 
allowance for the stations weight can be changed in line 136, or the distribution of the weights
can be changed in line 158 (ex: linear vs logarithmic).
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression

import arcpy
from arcpy import env
from arcpy.sa import *

arcpy.env.workspace = r"X:\MGIS\Capstone\plot_data\plot_data.gdb"

import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.crs import CRS

# This function resamples the input rasters using predefined resolutions and extents
def resample_raster(input_path, new_shape, new_bounds):
   with rasterio.open(input_path, 'r+') as src:
       #src.crs = CRS.from_epsg(32617)
       # Calculate new resolution
       new_height, new_width = new_shape
       new_transform = Affine.translation(new_bounds[0], new_bounds[3]) * \
                       Affine.scale((new_bounds[2] - new_bounds[0]) / new_width,
                                    (new_bounds[1] - new_bounds[3]) / new_height)


       # Resample the raster
       data = src.read(
           out_shape=(src.count, new_height, new_width),
           resampling=Resampling.bilinear
       )


       # Update metadata
       profile = src.profile
       profile.update(transform=new_transform, width=new_width, height=new_height,
                      bounds=new_bounds)


       # Write to the output file
       # with rasterio.open(output_path, 'w', **profile) as dst:
       #     dst.write(data)
     
       return data[0,:,:]
   

# First, read in the values for each source 
#raster_list = arcpy.ListRasters("*")
input_mrms = r"X:\MGIS\Capstone\exported_rasters\mrms_precip_12z_20230829.tif"  #_20230829, _20230830, _20230831
input_wu = r"X:\MGIS\Capstone\exported_rasters\idw_12z_wu_20230829.tif" #_20230829, _20230830, _20230831
input_acis = r"X:\MGIS\Capstone\exported_rasters\idw_cocorahs_coop_20230829.tif" #_20230829, _20230830, _20230831
input_prism = r"X:\MGIS\Capstone\exported_rasters\prism_20230829_12z_in.tif" #_20230829, _20230830, _20230831



# Define variables for the resampling function to run
# output_raster = "output.tif" # Could export the raster with this line uncommented (need to add output_raster to function params)
new_grid_size = (13, 17)  # new shape (height, width)
#new_bounds = (319310.965464, 407754.060227, 3059085.267864, 3126718.222682)  # new bounds (left, right, bottom, top) -northings/eastings
new_bounds = (-82.8314, -81.9404, 27.6439, 28.2631)  # new bounds (left, right, bottom, top) -coordinates


mrms_values = resample_raster(input_mrms, new_grid_size, new_bounds)

plt.figure()
plt.imshow(np.ma.masked_where(mrms_values<0,mrms_values,np.nan),cmap='terrain')
plt.colorbar()
plt.title('This is resampled MRMS data')
plt.show()

wu_values = resample_raster(input_wu, new_grid_size, new_bounds)

plt.figure()
plt.imshow(np.ma.masked_where(wu_values<0,wu_values,np.nan),cmap='terrain')
plt.colorbar()
plt.title('This is resampled WU data')
plt.show()

acis_values = resample_raster(input_acis, new_grid_size, new_bounds)

plt.figure()
plt.imshow(np.ma.masked_where(acis_values<0,acis_values,np.nan),cmap='terrain')
plt.colorbar()
plt.title('This is resampled ACIS data')
plt.show()

prism_values = resample_raster(input_prism, new_grid_size, new_bounds)

plt.figure()
plt.imshow(np.ma.masked_where(prism_values<0,prism_values,np.nan),cmap='terrain')
plt.colorbar()
plt.title('This is resampled PRISM data')
plt.show()

# Now, moving on to the weighting scheme work
wu_reclass = Raster("Reclass_wu")
acis_reclass = Raster("Reclass_acis_0829") #_0829, _0830, _0831
combo_reclass = Raster("Reclass_combo_0829") #_0829, _0830, _0831


"""
wu_desc = arcpy.Describe(wu_reclass)
acis_desc = arcpy.Describe(acis_reclass)
combo_desc = arcpy.Describe(combo_reclass)
print(wu_desc.meanCellWidth)
print(wu_desc.meanCellHeight)
print(acis_desc.meanCellWidth)
print(acis_desc.meanCellHeight)
print(combo_desc.meanCellWidth)
print(combo_desc.meanCellHeight)
"""

# First, decide total percentage of MRMS vs stations weighting

combo_weight = np.linspace(0, .45, num=10)
#print(combo_weight)

combo_reclass_arr = arcpy.RasterToNumPyArray(combo_reclass)
for i in range(10):
    combo_reclass_arr = np.where(combo_reclass_arr == i+1, combo_weight[i], combo_reclass_arr)

mrms_weight = 1-combo_reclass_arr
"""
plt.imshow(combo_reclass_arr)
plt.show()
plt.imshow(mrms_weight)
plt.colorbar()
plt.show()
"""


# Create list representing all the possible difference values from the reclass rasters (-9 to 9)
# ranging 0 to 100% so that they can later be scaled into the combo reclass weight

diff_list = list(itertools.chain(range(-9, 9+1)))
new_weights = np.linspace(0, 1, num=19)
#print(diff_list)
#print(new_weights)

wu_reclass_arr = arcpy.RasterToNumPyArray(wu_reclass)
wu_reclass_arr = wu_reclass_arr.astype(np.int16)

acis_reclass_arr = arcpy.RasterToNumPyArray(acis_reclass)
acis_reclass_arr = acis_reclass_arr.astype(np.int16)


diff_arr_wu = wu_reclass_arr - acis_reclass_arr
diff_arr_acis = acis_reclass_arr - wu_reclass_arr


for i in diff_list:
    #print(i)
    weight_index = diff_list.index(i)
    #print(weight_index)
    diff_arr_wu = np.where(diff_arr_wu == i, new_weights[weight_index], diff_arr_wu)


#print(diff_list[::-1])
for i in diff_list[::-1]:
    #print(i)
    weight_index = diff_list.index(i)
    #print(weight_index)
    diff_arr_acis = np.where(diff_arr_acis == i, new_weights[weight_index], diff_arr_acis)
"""
sanity_check1 = diff_arr_acis + diff_arr_wu
plt.imshow(sanity_check1)
plt.colorbar()
plt.show()
"""

# Now we need to scale the above weights to make them work with the confines of the 
# combo reclass array (i.e. the allowance we've given to the stations vs MRMS)
    
final_wu_weight = np.multiply(diff_arr_wu, combo_reclass_arr)
final_acis_weight = np.multiply(diff_arr_acis, combo_reclass_arr)

"""
sanity_check2 = (final_wu_weight + final_acis_weight) - combo_reclass_arr
plt.imshow(sanity_check2)
plt.colorbar()
plt.show()

sanity_check3 = final_wu_weight + final_acis_weight + mrms_weight
plt.imshow(sanity_check3)
plt.colorbar()
plt.show()
"""

# Apply the weighting scheme to make the new output
new_model = (mrms_weight * mrms_values) + (final_wu_weight * wu_values) + (final_acis_weight * acis_values)
new_model = np.ma.masked_where(prism_values == -99, new_model, np.nan)

plt.figure()
plt.imshow(new_model)
plt.colorbar()
plt.title('New MRMS output using WU and COCORAHS/COOP stations')
plt.show()

# Calculate initial difference between MRMS and PRISM to later compare new model results
initial_diff = mrms_values - prism_values
initial_diff = np.ma.masked_where(prism_values == -99, initial_diff, np.nan)

plt.figure()
plt.imshow(initial_diff)
plt.colorbar()
plt.title('This is initial MRMS - PRISM')

# Take the difference of the new MRMS model and PRISM
new_diff = new_model - prism_values

plt.figure()
plt.imshow(new_diff)
plt.colorbar()
plt.title('This is the new MRMS - PRISM')
plt.show()

# Produce new scatter plots to display the correlations (or lack thereof) between MRMS and PRISM
flat_mi = mrms_values.flatten()
flat_mn = new_model.flatten()
flat_p = prism_values.flatten()

invalid = -99
valid_indices = (flat_p != invalid) & (flat_mi != invalid)

flat_mi = flat_mi[valid_indices]
flat_mn = flat_mn[valid_indices]
flat_p = flat_p[valid_indices]

max = max([max(flat_mi), max(flat_mn), max(flat_p)])
print(max)


# For initial difference
plt.figure(figsize=(6,6))
plt.scatter(flat_p, flat_mi)
plt.xlim(0,max)
plt.ylim(0,max)
plt.title('Initial MRMS and PRISM comparison for 20230829')
plt.xlabel('PRISM (in)')
plt.ylabel('MRMS (in)')
plt.plot([0,max],[0,max],label='1:1', color='black')

# Masked array values must first be removed before making a scatter plot for R2 calculation
x = flat_p.reshape(-1, 1)
y = flat_mi

model = LinearRegression().fit(x, y)
r_squared1 = model.score(x, y)
print(r_squared1)
test = np.array([0,max])
plt.plot(test,[(model.coef_  * t) + model.intercept_ for t in test],
        label='Regression line: y = ' + str(round(model.coef_[0],3)) + 'x + ' + str(round(model.intercept_,3)), color='purple', linestyle='--')

plt.text(.05, max-max/15, 'R\u00B2 value = ' + str(round(r_squared1,3)), fontweight = 'bold', fontsize = 12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
#plt.savefig(outfold + 'scatter_' + sr1 + '_' + sr2 + pt_file[-12:-4] + '.png', dpi=300)


# For new difference
plt.figure(figsize=(6,6))
plt.scatter(flat_p, flat_mn)
plt.xlim(0,max)
plt.ylim(0,max)
plt.title('New MRMS (with WU and ACIS) and PRISM comparison for 20230829')
plt.xlabel('PRISM (in)')
plt.ylabel('MRMS (in)')
plt.plot([0,max],[0,max],label='1:1', color='black')

# Masked array values must first be removed before making a scatter plot for R2 calculation
x = flat_p.reshape(-1, 1)
y = flat_mn

model = LinearRegression().fit(x, y)
r_squared2 = model.score(x, y)
print(r_squared2)
test = np.array([0,max])
plt.plot(test,[(model.coef_  * t) + model.intercept_ for t in test],
        label='Regression line: y = ' + str(round(model.coef_[0],3)) + 'x + ' + str(round(model.intercept_,3)), color='purple', linestyle='--')

plt.text(.05, max-max/15, 'R\u00B2 value = ' + str(round(r_squared2,3)), fontweight = 'bold', fontsize = 12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
#plt.savefig(outfold + 'scatter_' + sr1 + '_' + sr2 + pt_file[-12:-4] + '.png', dpi=300)
plt.show()