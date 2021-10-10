import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os
import shap

'''
Script for visualizing the results of SHAP for a given ML model.
Requires SHAP values to be saved from SHAP.py.
Generates (a) a SHAP summary plot for prediction of CH4 permeability
and (b) a heatmap plot showing the top 10 most important chemical features
overall with respective impacts on 6 gas permeabilities.
'''

def plot(args):
    modeltype = args.modelname.split('_')[0]
    imputation = args.modelname.split('_')[1]
    features = args.modelname.split('_')[2]
    maindirectory = os.getcwd() + '\\models\\' + args.modelname
    
    X_df = pd.read_csv(os.getcwd() + '\\datasets\\datasetAX_' + features + '.csv')

    #obtain the proper Xscaler
    X = np.array(X_df)
    Xscaler = StandardScaler()
    X= Xscaler.fit_transform(X)

    os.chdir(maindirectory)
    shap_values = avg_shap_values = np.zeros((6, X.shape[0], X.shape[1]))
    for i in range(6):
        shap_values[i, :, :] = pd.read_csv('shap_' + str(i) + '.csv', header=None)

    #plot the 
    shap.summary_plot(shap_values[5], X, feature_names=X_df.columns, show=False)
    plt.show()

    wts = np.zeros((X_df.shape[1],6))
    for i in range(6):
        wts[:,i] = np.mean(np.abs(shap_values[i]),0)
    wts = pd.DataFrame(wts)
    wts['sum'] = np.sum(wts, axis=1)
    wts['id'] = X_df.columns
    ordered_wts = wts.sort_values('sum', ascending=False)
    
    #sns.set_theme(style='ticks', palette='bright', font='sans-serif', font_scale=1.2)
    #sns.set_theme(style='ticks', palette='Blues', font='sans-serif', font_scale=1.2)
    fig = plt.figure(figsize=(16,8)) 
    Columns = ['$He$','$H_2$','$O_2$','$N_2$','$CO_2$','$CH_4$']
    topwts = ordered_wts[:10]
    labels = topwts['id'].to_numpy().astype(int)
    sns.heatmap(ordered_wts.iloc[:10,:6], annot=True, xticklabels=Columns, yticklabels=labels,linewidths=2, cmap="Blues")
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='provide the name of the folder in the "model" folder')        
    parsed_args = parser.parse_args()

    plot(parsed_args)