"""
2/16/2024
Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script takes a list of rasters of NEXRAD Level III One-Hr Precip data and adds them up into a 24 hour raster from 0Z to 0Z.
"""

import numpy as np
import arcpy
import matplotlib.pyplot as plt

arcpy.env.workspace = "X:/MGIS/Capstone/data/nexrad/nexrad_20230831_12z_24hr"
arcpy.env.overwriteOutput = True

# Set up raster list - may need to edit cell size if different data is used.
rasters = arcpy.ListRasters()
#srs = arcpy.Describe(rasters[1]).spatialReference
x_cell_size = 0.007477770236569831
y_cell_size = 0.007486478073604091
lowerLeft = arcpy.Point(arcpy.Raster(rasters[1]).extent.XMin, arcpy.Raster(rasters[1]).extent.YMin)

new_arr = []

# Go through all rasters and convert them to numpy arrays
for r in rasters:
    arr = arcpy.RasterToNumPyArray(r)
    arr = np.nan_to_num(arr)
    plt.imshow(arr)
    plt.colorbar()
    plt.show()
    new_arr.append(arr)

# Add all 24 arrays one by one until 24 hr raster is complete
final_arr = np.zeros((558,800))
for a in new_arr:
    final_arr = np.add(final_arr, a)
    plt.imshow(final_arr)
    plt.colorbar()
    plt.show()

# Export and save.  Will need to set CRS in ArcPro
new_ras = arcpy.NumPyArrayToRaster(final_arr, lowerLeft, x_cell_size, y_cell_size)
new_ras.save("X:/MGIS/Capstone/plot_data/plot_data.gdb/nexrad_24hr_20230831")

