import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
tf.keras.backend.set_floatx('float64')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import os
import shap
from DNN_functions import nanmean_squared_error

'''
Script to obtain SHAP values from ML models.
Requires the trained models as saved by step3_train.py.
Outputs 6 .csv files in the order ['He','H2','O2','N2','CO2','CH4'] where each file displays the SHAP weights for a single permeability prediction.
The rows correspond to samples and the columns correspond to chemical features.
'''

def SHAP(args):

    modeltype = args.modelname.split('_')[0]
    imputation = args.modelname.split('_')[1]
    features = args.modelname.split('_')[2]

    maindirectory = os.getcwd() + '/models/' + args.modelname
    
    X_df = pd.read_csv(os.getcwd() + '/datasets/datasetAX_' + features + '.csv')
    os.chdir(maindirectory)

    #obtain the proper Xscaler
    X = np.array(X_df)
    if features == 'desc':
        Xscaler = StandardScaler()
        X= Xscaler.fit_transform(X)

    if modeltype == 'RF':
        filename = args.modelname + '.sav'
        model = pickle.load(open(filename, 'rb'))

        background = X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background)
    
    if modeltype == 'DNN':
        max_index = 15

        background = X
        avg_shap_values = np.zeros((6, X.shape[0], X.shape[1]))
        for i in range(max_index+1):
            print('Calculating SHAP values ' + str(i) + '/' + str(max_index+1))
            directory = maindirectory + '/DNN_' + str(i)
            model = tf.keras.models.load_model(directory, custom_objects={'nanmean_squared_error': nanmean_squared_error})
            e = shap.explainers._deep.Deep(model, background)
            shap_values = e.shap_values(background)
            avg_shap_values = avg_shap_values + shap_values

        avg_shap_values = avg_shap_values/(max_index+1)
        shap_values = avg_shap_values
    
    Columns = ['H2','He','O2','N2','CO2','CH4']
    for i in range(len(shap_values)):
        gas = Columns[i]
        np.savetxt('shap_' + str(i) + '.csv', shap_values[i], delimiter=",")
    print('SHAP values saved to ' + maindirectory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='provide the name of the folder in the "model" folder')        
    parsed_args = parser.parse_args()

    SHAP(parsed_args)