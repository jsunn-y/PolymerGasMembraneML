import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

'''
Script to visualize permeability data in the O2/N2 and CO2/CH4 Robeson spaces.
Data must be a csv file with 6 columns in the order of ['H2','He','O2','N2','CO2','CH4'] permeabilities.
Update the filelist list with paths to each of the .csv files that are desired.
Works with the .csv outputs of screen.py and train.py.
'''

#change the paths to the files that are desired to be plotted
filelist= ['models/RF_BLR_fing/Y_train.csv', 'models/RF_BLR_fing/Y_pred_datasetD.csv']

sns.set_palette("colorblind")

#plot permeability values on a Robeson plot
fig = plt.figure(figsize=(12,4))

for i, dataset in enumerate(filelist):
    #read in the permeability data for each dataset
    Y_pred = pd.read_csv(dataset, header=None)
    Y_pred = np.array(Y_pred)

    #O2/N2 separations
    plt.subplot(1,2,1)
    permeability = Y_pred[:,-4]
    selectivity = Y_pred[:,-4] - Y_pred[:,-3]
    plt.plot(permeability, selectivity, '.', alpha = 0.2)

    #CO2/CH4 separations
    plt.subplot(1,2,2)
    permeability = Y_pred[:,-2]
    selectivity = Y_pred[:,-2] - Y_pred[:,-1]
    plt.plot(permeability, selectivity, '.', alpha=0.2)
    
#format the plot and add Robeson upper bounds
plt.subplot(1,2,1)
xmin = -4
xmax = 5
plt.xlim([xmin, xmax])
plt.ylim([-1, 2])
plt.plot([xmin, xmax], [np.log10(9.2008)-0.1724*xmin, np.log10(9.2008)-0.1724*xmax], '-k') #1991 upper bound
plt.plot([xmin, xmax], [np.log10(12.148)-0.1765*xmin, np.log10(12.148)-0.1765*xmax], '--k') #2008 upper bound
plt.plot([xmin, xmax], [np.log10(18.50)-0.1754*xmin, np.log10(18.50)-0.1754*xmax], ':k') #2015 upper bound
plt.title("O2/N2 Separations")
plt.legend(filelist)

plt.subplot(1,2,2)
xmin = -2
xmax =5
plt.xlim([xmin, xmax])
plt.ylim([-2, 4])
plt.plot([xmin, xmax], [np.log10(197.81)-0.3807*xmin, np.log10(197.81)-0.3807*xmax], '-k') #1991 upper bound
plt.plot([xmin, xmax], [np.log10(357.33)-0.3794*xmin, np.log10(357.33)-0.3794*xmax], '--k') #2008 upper bound
plt.plot([xmin, xmax], [np.log10(1155.60)-0.4165*xmin, np.log10(1155.60)-0.4165*xmax], ':k') #2019 upper bound
plt.title('CO2/CH4 Separations')

plt.show()
