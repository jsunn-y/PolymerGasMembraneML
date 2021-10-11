import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
tf.keras.backend.set_floatx('float64')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import pickle
import argparse
import os
from DNN_functions import nanmean_squared_error, evaluate_model, ensemble_predictions

'''
Script to predict the permeabilities of polymers with known SMILES strings
but unkown permeability, based on trained ML models.
Requires the trained models as saved by step3_train.py. Relevant chemical features
of the screenign dataset must also be saved by step2_generate_Xfeatures.py.
Outputs .csv files with predicted permeabilities of six gases in column order ['H2','He','O2','N2','CO2','CH4'].
'''

def screen(args):
    modeltype = args.modelname.split('_')[0]
    imputation = args.modelname.split('_')[1]
    features = args.modelname.split('_')[2]

    maindirectory = os.getcwd() + '\\models\\' + args.modelname

    DatasetA_Smiles_P = pd.read_csv("datasets\datasetA_imputed_all.csv")
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles').mean().reset_index()

    if imputation == 'BLR':
        Y = DatasetA_grouped.iloc[:,-12:-6]
    if imputation == 'ERT':
        Y = DatasetA_grouped.iloc[:,-6:]
    
    Xstandard = pd.read_csv(os.getcwd() + '\\datasets\\datasetAX_' + features + '.csv')
    X_pred = pd.read_csv(os.getcwd() + '\\datasets\\' + args.dataset + 'X_' + features + '.csv')
    os.chdir(maindirectory)

    #obtain the proper Xscaler
    Xstandard = np.array(Xstandard)
    Xscaler = StandardScaler()
    Xscaler.fit(Xstandard)
    X_pred= Xscaler.transform(X_pred)

    #obatin the proper Yscaler
    Y = np.array(Y)
    Yscaler = StandardScaler()
    Yscaler.fit(Y)

    if modeltype == 'RF':
        filename = args.modelname + '.sav'
        model = pickle.load(open(filename, 'rb'))

        Y_pred = model.predict(X_pred)
        Y_pred = Yscaler.inverse_transform(Y_pred)
        filename = 'Y_pred_' + args.dataset + '.csv'
        np.savetxt(filename, Y_pred, delimiter=",")
        print('Predictions saved to ' + maindirectory + '\\' + filename)
    
    if modeltype == 'DNN':
        folders = os.listdir(maindirectory)
        indices = []
        for name in folders:
            if os.path.isdir(name):
                indices.append(int(name.split('_')[1]))
        max_index = max(indices)

        models = list()
        for i in range(max_index+1):
            directory = maindirectory + '/DNN_' + str(i)
            models.append(tf.keras.models.load_model(directory, custom_objects={'nanmean_squared_error': nanmean_squared_error}))

        Y_pred, Y_var = ensemble_predictions(models, X_pred)
        Y_pred = Yscaler.inverse_transform(Y_pred)
        filename = 'Y_pred_' + args.dataset + '.csv'
        np.savetxt(filename, Y_pred, delimiter=",")
        print('Predictions saved to ' + maindirectory + '\\' + filename)

#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, required = True, 
    	help='provide the name of the folder in the "model" folder')        
    parser.add_argument('--dataset', type=str, required = True, 
    	help='choose the file name of one of the descriptor of fingerprint datasets (B,C,D) to use as inputs in the ML model')

    parsed_args = parser.parse_args()

    screen(parsed_args)