import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os
import shap

'''
Script for visualizing the results of SHAP for a given ML model.
Requires SHAP values to be saved from SHAP.py.
Generates (a) a SHAP summary plot for prediction of each permeability
and (b) a 3D bar graph showing the top 12 most important chemical features
overall with respective impacts on 6 gas permeabilities.
'''

def plot(args):
    modeltype = args.modelname.split('_')[0]
    imputation = args.modelname.split('_')[1]
    features = args.modelname.split('_')[2]
    maindirectory = os.getcwd() + '/models/' + args.modelname
    
    X_df = pd.read_csv(os.getcwd() + '/datasets/datasetAX_' + features + '.csv')

    X = np.array(X_df)
    Xscaler = StandardScaler()
    X= Xscaler.fit_transform(X)

    os.chdir(maindirectory)
    shap_values = avg_shap_values = np.zeros((6, X.shape[0], X.shape[1]))
    for i in range(6):
        shap_values[i, :, :] = pd.read_csv('shap_' + str(i) + '.csv', header=None)

    for i in range(6):
        fig = plt.figure()
        shap.summary_plot(shap_values[i], X, show=False)
    plt.show()

    wts = np.zeros((X_df.shape[1],6))
    for i in range(6):
        wts[:,i] = np.mean(np.abs(shap_values[i]),0)
    wts = pd.DataFrame(wts)
    wts['sum'] = np.sum(wts, axis=1)
    wts['id'] = X_df.columns
    ordered_wts = wts.sort_values('sum', ascending=False)
    top_ordered_wts = ordered_wts.iloc[:12,:]
    top_ordered_wts = top_ordered_wts.rename(top_ordered_wts['id'])
    fig = plt.figure(figsize=(20, 5))
    top_ordered_wts.iloc[:,:6].plot(kind='bar', colormap='Set2')
    plt.ylabel('Avg Magnitude of SHAP Value')
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='provide the name of the folder in the "model" folder')        
    parsed_args = parser.parse_args()

    plot(parsed_args)