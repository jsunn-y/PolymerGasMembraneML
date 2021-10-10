import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import r2_score

'''
Script to visualize the performance of a given ML model.
Requires outputs of model training in train.py.
Generates plots of actual permeability against predicted permeability for 6 gases.
'''

def plot(args):
    #load in the data from the model training
    os.chdir(os.getcwd() + '\\models\\' + args.directory)
    Y_train=pd.read_csv('Y_train.csv', header=None)
    Y_train = Y_train.to_numpy()
    Y_pred_train=pd.read_csv('Y_pred_train.csv', header=None)
    Y_pred_train = Y_pred_train.to_numpy()
    Y_test=pd.read_csv('Y_test.csv', header=None)
    Y_test = Y_test.to_numpy()
    Y_pred_test=pd.read_csv('Y_pred_test.csv', header=None)
    Y_pred_test = Y_pred_test.to_numpy()

    #plot the performance of the model for six gases
    Columns = ['H2','He','O2','N2','CO2','CH4']

    fig = plt.figure(figsize=(12,8))
    for i in range(6):
        ax=plt.subplot(2, 3, i + 1)
        index = i #adjust so that we look at the last 4 gases
        plt.plot(Y_train[:,index], Y_pred_train[:,index], '.', color='tab:purple')#, alpha=0.8, label="Train R^2: {}".format(r2_score(y_train, Y_pred_train)))
        plt.plot(Y_test[:,index], Y_pred_test[:,index], 'g.')#, alpha=0.8, label="Test R^2: {}".format(r2_score(y_test, Y_pred_test)))
        plt.legend(['Train', 'Test'], loc = 'best')
        plt.xlabel(Columns[i]) 
        plt.ylabel("Predicted value")
        x0, x1 = min(Y_train[:,index]), max(Y_train[:,index])
        length = x1 - x0
        x_start, x_end = x0-0.1*length, x1+0.1*length
        plt.xlim([x_start, x_end])
        plt.ylim([x_start, x_end])
        
        # the unit line
        plt.plot(np.arange(x_start, x_end, 0.01*length),
        np.arange(x_start, x_end, 0.01*length), '-', color='tab:gray')
        plt.text(x_end - 0.7*length, x_start + 0.15*length, "$Train R^2={:.2f}$".format(r2_score(Y_train[:,index], Y_pred_train[:,index])))
        plt.text(x_end - 0.7*length, x_start + 0.05*length, "$Test R^2={:.2f}$".format(r2_score(Y_test[:,index], Y_pred_test[:,index])))
    
    plt.show()

#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='give the folder name that contains the trained model with no backslashes, should be within the "models" folder')
    
    parsed_args = parser.parse_args()
    plot(parsed_args)