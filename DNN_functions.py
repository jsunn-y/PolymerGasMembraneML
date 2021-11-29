import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
'''
Useful functions in the training and evaluation of the DNN ensemble
'''
#user defined mean squared errors that ignores NaN values, used as the loss function when training on non-imputed data with missing entries
def nanmean_squared_error(y_true, Y_pred):
    Y_pred = ops.convert_to_tensor_v2(Y_pred)
    y_true = math_ops.cast(y_true, Y_pred.dtype)
    residuals = (y_true - Y_pred)
    residuals_no_nan = tf.where(tf.math.is_nan(residuals), tf.zeros_like(residuals), residuals)
    sum_residuals = tf.reduce_sum(math_ops.squared_difference(residuals_no_nan , 0),-1) / tf.reduce_sum(tf.cast(~tf.math.is_nan(y_true), tf.float64),-1)
    return sum_residuals

# evaluate a single DNN for regression 
def evaluate_model(trainX, trainy, testX, testy):
    model = keras.models.Sequential()
    model.add(Dense(units = 64,activation='relu')) #input_dim = X.shape[1]
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dense(units = 32, activation='relu'))
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units = 8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units = 6))

    # model.summary()
    #model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.compile(loss = nanmean_squared_error, optimizer = 'adam')
    history = model.fit((trainX), trainy, epochs = 50, batch_size = 64, validation_data = ((testX), testy), verbose=0)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(loss)
    #plt.figure()
    #plt.plot(range(epochs), loss, marker = '.', label = 'loss')
    #plt.plot(range(epochs), val_loss, marker = '.', label = 'val_loss')
    #plt.grid()
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')

    # evaluate the model
    predy = model.predict((testX))
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:,i])
        test_r2[i] = r2_score(testy[:,i][flag1], predy[:,i][flag1], multioutput='raw_values')
    return model, test_r2

# make an ensemble prediction
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # mean of ensemble members
    predictions = np.mean(yhats, axis=0)
    variances = np.var(yhats, axis=0)
    #print(summed)
    return predictions, variances

# evaluate a specific number of members in an ensemble for the regression score (R2) and the variance of predictions
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat, variances = ensemble_predictions(subset, testX)
    avg_var = np.mean(variances, axis =0)
    
    # calculate R2
    test_r2 = np.zeros(6)
    for i in range(6):
        flag1 = ~np.isnan(testy[:,i])
        test_r2[i] = r2_score(testy[:,i][flag1], yhat[:,i][flag1], multioutput='raw_values')
    return test_r2, avg_var