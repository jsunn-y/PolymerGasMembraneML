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
from DNN_functions import nanmean_squared_error, evaluate_model, ensemble_predictions, evaluate_n_members


'''
Script to train the ML models used in this work.
Input features can be chemical descriptors ("desc") or Morgan fingerprints ("fing").
Imputation can be Bayesian linear regression ("BLR") or extremely randomized trees ("ERT").
Model can either be random forest ("RF") or ensemble of deep neural networks ("DNN").
Outputs the saved model into the model folder, along with Y_train, Y_test, Y_pred_train, and Y_pred_test as .csv files.
Each .csv file contains columns as gas permeability in the order ['He','H2','O2','N2','CO2','CH4'] and rows as samples.
'''

def train(args):

    #read in the training data
    DatasetA_Smiles_P = pd.read_csv("datasets/datasetA_imputed_all.csv")
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles').mean().reset_index()

    if args.imputation == 'BLR':
        Y = DatasetA_grouped.iloc[:,-12:-6]
    if args.imputation == 'ERT':
        Y = DatasetA_grouped.iloc[:,-6:]
    
    #normalize Y
    Y = np.array(Y)
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)

    if args.features == 'desc':
        X = pd.read_csv('datasets/datasetAX_desc.csv')

        #normalize X
        X = np.array(X)
        Xscaler = StandardScaler()
        X = Xscaler.fit_transform(X)
    if args.features == 'fing': 
        X = pd.read_csv('datasets/datasetAX_fing.csv')
        X = np.array(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    if args.model == 'RF':
        model = RandomForestRegressor(n_estimators=200, max_depth = 10, bootstrap = True, max_features = 'sqrt')
        print('Training random forest model...')
        history = model.fit(X_train, Y_train)
        Y_train = scaler.inverse_transform(Y_train)
        Y_test = scaler.inverse_transform(Y_test)

        Y_pred_train = model.predict((X_train))
        Y_pred_train = scaler.inverse_transform(Y_pred_train)
        Y_pred_test = model.predict((X_test))
        Y_pred_test = scaler.inverse_transform(Y_pred_test)

        maindirectory = os.getcwd() + '/models/RF_' + args.imputation + '_' + args.features
        if  not os.path.exists(maindirectory):
            os.mkdir(maindirectory)
        filename = maindirectory + '/RF_' + args.imputation + '_' + args.features + '.sav'
        pickle.dump(model, open(filename, 'wb'))

        os.chdir(maindirectory)
        np.savetxt('Y_train.csv', Y_train, delimiter=",")
        np.savetxt('Y_test.csv', Y_test, delimiter=",")
        np.savetxt('Y_pred_train.csv', Y_pred_train, delimiter=",")
        np.savetxt('Y_pred_test.csv', Y_pred_test, delimiter=",")

        print('Model saved to ' + filename)

    if args.model == 'DNN':
        newX = X_test
        newy = Y_test

        n_splits = 16
        scores, members = list(), list()
        print('Training DNN model...')
        for _ in range(n_splits):
            # select indexes
            ix = [i for i in range(len(X))]
            train_ix = resample(ix, replace=True, n_samples=round(X.shape[0]*0.8)) #Bootstrap 80% of the training set for each model
            test_ix = [x for x in ix if x not in train_ix]
            # select data
            trainX, trainy = X[train_ix], Y[train_ix]
            testX, testy = X[test_ix], Y[test_ix]
            # evaluate model
            model, test_r2 = evaluate_model(trainX, trainy, testX, testy)
            print(test_r2)
            filename = 'DNN_fingerprint_Loss_' + str(_) + '.eps'
            #plt.savefig(filename, bbox_inches='tight')
            scores.append(test_r2)
            members.append(model)
        # summarize performance for each model
        print('Mean R2: ' + str(np.mean(scores, axis = 0)))
        print('Estimated Std of R2: ' + str(np.std(scores, axis = 0)))

        # evaluate different numbers of ensembles on hold out set
        single_scores, ensemble_scores = list(), list()
        for i in range(n_splits+1, n_splits+1):
            ensemble_score, ensemble_variance = evaluate_n_members(members, i, newX, newy)
            print(str(i) + ' Ensemble R2s: ' + str(ensemble_score))
            print(str(i) + ' Ensemble pred var: ' + str(ensemble_variance))
            print(' ')

        Y_pred_train, var = ensemble_predictions(members, X_train)
        Y_pred_train = scaler.inverse_transform(Y_pred_train)
        Y_pred_test, var = ensemble_predictions(members, newX)
        Y_pred_test = scaler.inverse_transform(Y_pred_test)
        Y_train = scaler.inverse_transform(Y_train)
        Y_test = scaler.inverse_transform(newy)

        #save current models
        print('Saving models...')
        for count, model in enumerate(members):
            directory = os.getcwd() + '/models/DNN_' + args.imputation + '_' + args.features + '/DNN_' + str(count)
            model.save(directory)
            maindirectory = os.getcwd() + '/models/DNN_' + args.imputation + '_' + args.features
        
        os.chdir(maindirectory)
        np.savetxt('Y_train.csv', Y_train, delimiter=",")
        np.savetxt('Y_test.csv', Y_test, delimiter=",")
        np.savetxt('Y_pred_train.csv', Y_pred_train, delimiter=",")
        np.savetxt('Y_pred_test.csv', Y_pred_test, delimiter=",")
        print('Model saved to ' + maindirectory)


#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required = True, 
    	help='Choose either "desc" for chemical descriptors or "fing" for fingerprints as the input features')
    parser.add_argument('--imputation', type=str, required = True, 
    	help='Choose either "BLR" for bayesian linear regression or "ERT" for etremely randomized trees for the imputation method')        
    parser.add_argument('--model', type=str, required = True, 
    	help='choose either "RF" for random forest or "DNN" for the ensemble of deep neural networks for model training')

    parsed_args = parser.parse_args()

    train(parsed_args)