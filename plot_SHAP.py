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

    #obtain the proper Xscaler
    X = np.array(X_df)
    Xscaler = StandardScaler()
    X= Xscaler.fit_transform(X)

    os.chdir(maindirectory)
    shap_values = avg_shap_values = np.zeros((6, X.shape[0], X.shape[1]))
    for i in range(6):
        shap_values[i, :, :] = pd.read_csv('shap_' + str(i) + '.csv', header=None)

    #plot the shap summary plot
    for i in range(6):
        fig = plt.figure()
        shap.summary_plot(shap_values[i], X, feature_names= np.array(['101', '107', '109','164', '79', '16', '15', '75', '116', '14']), show=False)
        plt.savefig('shap_' + str(i) + '.eps')
    plt.show()

    wts = np.zeros((X_df.shape[1],6))
    for i in range(6):
        wts[:,i] = np.mean(np.abs(shap_values[i]),0)
    wts = pd.DataFrame(wts)
    wts['sum'] = np.sum(wts, axis=1)
    wts['id'] = X_df.columns
    ordered_wts = wts.sort_values('sum', ascending=False)
    top_ordered_wts = ordered_wts.iloc[:12,:]

    result = np.array(top_ordered_wts.iloc[:,:6]).transpose()
    colors = ['r','b','g','y','b','p']
    fig=plt.figure(figsize=(8, 8), dpi=250)
    ax1=fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Feature ID', labelpad=10)
    ax1.set_ylabel('Gas Type', labelpad=10)
    ax1.set_zlabel('SHAP Importance')

    xlabels = top_ordered_wts['id']
    xpos = np.arange(xlabels.shape[0])
    ylabels = np.array(['$He$','$H_2$','$O_2$','$N_2$','$CO_2$','$CH_4$'])
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos=result
    zpos = zpos.ravel()

    dx=0.5
    dy=0.5
    dz=zpos

    ax1.w_xaxis.set_ticks(xpos - dx*2)
    ax1.w_xaxis.set_ticklabels(xlabels, rotation=45)

    ax1.w_yaxis.set_ticks(ypos  + 2*dy)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='provide the name of the folder in the "model" folder')        
    parsed_args = parser.parse_args()

    plot(parsed_args)