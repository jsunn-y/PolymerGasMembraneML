import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

'''
Script to visualize permeability data in the O2/N2, CO2/CH4, CO2/N2, and H2/CO2 Robeson spaces.
Data must be a csv file with 6 columns in the order of ['He','H2','O2','N2','CO2','CH4'] permeabilities.
Update the filelist list with paths to each of the .csv files that are desired.
Works with the .csv outputs of screen.py and train.py.
'''

#change the paths to the files that are desired to be plotted
filelist= ['models/DNN_BLR_fing/Y_pred_datasetBX_fing.csv', 'models/DNN_BLR_fing/Y_pred_datasetCX_fing_0.csv', 'models/DNN_BLR_fing2/Y_pred_datasetDX_fing.csv', 'models/RF_BLR_fing/Y_train.csv']

sns.set_palette("colorblind")

#plot permeability values on a Robeson plot
fig = plt.figure(figsize=(12,8))

for i, dataset in enumerate(filelist[:-1]):
    #read in the permeability data for each dataset
    Y_pred = pd.read_csv(dataset, header=None)
    Y_pred = np.array(Y_pred)

    #O2/N2 separations
    plt.subplot(2,2,1)
    permeability = Y_pred[:,-4]
    selectivity = Y_pred[:,-4] - Y_pred[:,-3]
    plt.plot(permeability, selectivity, '.', alpha = 0.2)

    #CO2/CH4 separations
    plt.subplot(2,2,2)
    permeability = Y_pred[:,-2]
    selectivity = Y_pred[:,-2] - Y_pred[:,-1]
    plt.plot(permeability, selectivity, '.', alpha=0.2)

    #CO2/N2 separations
    plt.subplot(2,2,3)
    permeability = Y_pred[:,-2]
    selectivity = Y_pred[:,-2] - Y_pred[:,-3]
    plt.plot(permeability, selectivity, '.', alpha=0.2)

    #H2/CO2 separations
    plt.subplot(2,2,4)
    permeability = Y_pred[:,-5]
    selectivity = Y_pred[:,-5] - Y_pred[:,-2]
    plt.plot(permeability, selectivity, '.', alpha=0.2)

#read in the permeability data for each dataset
Y_pred = pd.read_csv(filelist[-1], header=None)
Y_pred = np.array(Y_pred)

#O2/N2 separations
plt.subplot(2,2,1)
permeability = Y_pred[:,-4]
selectivity = Y_pred[:,-4] - Y_pred[:,-3]
plt.plot(permeability, selectivity, '.', markersize=3, color='k')

#CO2/CH4 separations
plt.subplot(2,2,2)
permeability = Y_pred[:,-2]
selectivity = Y_pred[:,-2] - Y_pred[:,-1]
plt.plot(permeability, selectivity, '.', markersize=3, color='k')

#CO2/N2 separations
plt.subplot(2,2,3)
permeability = Y_pred[:,-2]
selectivity = Y_pred[:,-2] - Y_pred[:,-3]
plt.plot(permeability, selectivity, '.', markersize=3, color='k')

#H2/CO2 separations
plt.subplot(2,2,4)
permeability = Y_pred[:,-5]
selectivity = Y_pred[:,-5] - Y_pred[:,-2]
plt.plot(permeability, selectivity, '.', markersize=3, color='k')
    
#format the plot and add Robeson upper bounds
plt.subplot(2,2,1)
xmin = -4
xmax = 7
plt.xlim([xmin, xmax])
plt.ylim([-1, 2])
plt.plot([xmin, xmax], [np.log10(9.2008)-0.1724*xmin, np.log10(9.2008)-0.1724*xmax], '-k') #1991 upper bound
plt.plot([xmin, xmax], [np.log10(12.148)-0.1765*xmin, np.log10(12.148)-0.1765*xmax], '--k') #2008 upper bound
plt.plot([xmin, xmax], [np.log10(18.50)-0.1754*xmin, np.log10(18.50)-0.1754*xmax], ':k') #2015 upper bound
plt.title("O2/N2 Separations")
#plt.legend(filelist)

plt.subplot(2,2,2)
xmin = -2
xmax =7
plt.xlim([xmin, xmax])
plt.ylim([-2, 4])
plt.plot([xmin, xmax], [np.log10(197.81)-0.3807*xmin, np.log10(197.81)-0.3807*xmax], '-k') #1991 upper bound
plt.plot([xmin, xmax], [np.log10(357.33)-0.3794*xmin, np.log10(357.33)-0.3794*xmax], '--k') #2008 upper bound
plt.plot([xmin, xmax], [np.log10(1155.60)-0.4165*xmin, np.log10(1155.60)-0.4165*xmax], ':k') #2019 upper bound
plt.title('CO2/CH4 Separations')

plt.subplot(2,2,3)
xmin = -2
xmax =7
plt.xlim([xmin, xmax])
plt.ylim([-1, 3])
plt.plot([xmin, xmax], -1/2.888*np.array([-np.log10(30967000)+xmin, -np.log10(30967000)+xmax]), '--k') #2008 upper bound
plt.plot([xmin, xmax], -1/3.409*np.array([-np.log10(755.58e6)+xmin, -np.log10(755.58e6)+xmax]), ':k') #2019 upper bound
plt.title('CO2/N2 Separations')

plt.subplot(2,2,4)
xmin = -2
xmax =7
plt.xlim([xmin, xmax])
plt.ylim([-1.5, 2])
plt.plot([xmin, xmax], -1/1.9363*np.array([-np.log10(1200)+xmin, -np.log10(1200)+xmax]), '-k') #1991 upper bound
plt.plot([xmin, xmax], -1/2.302*np.array([-np.log10(4515)+xmin, -np.log10(4515)+xmax]), '--k') #2008 upper bound
plt.title('H2/CO2 Separations')

plt.show()
