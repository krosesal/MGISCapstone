"""
2/17/2024
Author: Kristina Preucil
krs5664@psu.edu
"""

"""
This script will create the analysis plots for the capstone project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Sources are 'asos', 'nexrad', 'mrms', 'prism', 'wu', 'acis'
# Max is by date: 0829 - 1.5, 0830 - 4.0, 0831 - 6.5

pt_file = "X:/MGIS/Capstone/data/all_compare_pts_20230831.xls"
max = 6.5
outfold = "X:/MGIS/Capstone/graphics/wu_scatters/"


# Function to create scatter plot, compute R2 value, and plot regression line
def scatplt(pt_file, sr1, sr2, max, outfold):
    df = pd.read_excel(pt_file)

    # Create scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(df[sr1], df[sr2])
    plt.xlim(0,max)
    plt.ylim(0,max)
    plt.title('Precipitation Source Comparison | ' + pt_file[-12:-8] + '-' + pt_file[-8:-6] + '-' + pt_file[-6:-4])
    plt.xlabel(sr1.upper() + ' (in)')
    plt.ylabel(sr2.upper() + ' (in)')
    plt.plot([0,max],[0,max],label='1:1', color='black')


    x = np.array(df[sr1]).reshape((-1, 1))
    y = np.array(df[sr2])
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    #print(r_squared)
    test = np.array([0,max])
    plt.plot(test,[(model.coef_  * t) + model.intercept_ for t in test],
            label='Regression line: y = ' + str(round(model.coef_[0],3)) + 'x + ' + str(round(model.intercept_,3)), color='purple', linestyle='--')

    plt.text(.05, max-max/15, 'R\u00B2 value = ' + str(round(r_squared,3)), fontweight = 'bold', fontsize = 12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(outfold + 'scatter_' + sr1 + '_' + sr2 + pt_file[-12:-4] + '.png', dpi=300)
    plt.show()


# Run using function to cover all comparisons.  Each plot will save.
#scatplt(pt_file, 'asos', 'nexrad', max, outfold)
scatplt(pt_file, 'wu', 'nexrad', max, outfold)
scatplt(pt_file, 'wu', 'mrms', max, outfold)
scatplt(pt_file, 'wu', 'prism', max, outfold)
scatplt(pt_file, 'wu', 'acis', max, outfold)

