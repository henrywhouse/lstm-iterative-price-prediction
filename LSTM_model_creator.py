#################################
#  LSTM MODEL CREATOR METHODS   #
#################################

##################### Imports ###########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

##################  Min-Max Scaling  ####################################

def mm_scale(df):
    dates = np.array(df.index)
    mm_sc = MinMaxScaler()
    data_sc = mm_sc.fit_transform(df)
    return dates, data_sc

###################  Train/Val/Test Split  ##############################

def tvt_split(dates, sc_arr, train_prop, val_prop):
    
    len_df = len(sc_arr)
    train_index = int(len_df*train_prop)
    val_index = int(len_df*(train_prop+val_prop))

    dates_train = dates[:train_index]
    dates_val = dates[train_index:val_index]
    dates_test = dates[val_index:]

    data_train = sc_arr[:train_index]
    data_val = sc_arr[train_index:val_index]
    data_test = sc_arr[val_index:]

    return dates_train, data_train, dates_val, data_val, dates_test, data_test

################  Convert Array to Windowed Array  #######################

def arr_to_xy(arr, look_back):
    
    x = []
    y = []
    for i in range(look_back,arr.shape[0]):
        x.append(arr[i-look_back:i])
        y.append(arr[i][-1])

    return  np.array(x), np.array(y)

##############  Build Model; Train Model; Make Predictions  ##############

def lstm_model_pred(x_train, y_train, x_val, y_val, x_test, epochs):

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2]), activation='tanh'))
    model.add(Dense(256))
    model.add(Dense(256))
    model.add(Dense(1))
    model.build()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae'])
    model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs, verbose=1)
    
    train_pred = model.predict(x_train).flatten()
    val_pred = model.predict(x_val).flatten()
    test_pred = model.predict(x_test).flatten()

    return train_pred, val_pred, test_pred

########################  Simple Backtest of Buy/Sell  ####################

def simple_backtest(look_back, dates_test, y_test, test_pred):
    
    df = pd.DataFrame({'Obs':y_test, 'Pred':test_pred})
    df.index = dates_test[look_back:]
    df['Strat'] = df['Pred'] > df['Obs'].shift(1)
    df['Strat'] = df['Strat'].dropna().astype(int)
    df['Long Daily Ret'] = df['Obs'].pct_change()
    df['Strat Daily Ret'] = df['Strat']*df['Obs'].pct_change()
    df['Long Cum Ret'] = (df['Long Daily Ret']+1).cumprod() - 1
    df['Strat Cum Ret'] = (df['Strat Daily Ret']+1).cumprod() - 1

    return df

#################   VISUALIZE METHODS   ##################################

def tvt_vis(dates_train, data_train, dates_val, data_val, dates_test, data_test):
    
    plt.plot(dates_train, data_train)
    plt.plot(dates_val, data_val)
    plt.plot(dates_test, data_test)
    plt.legend(['Train', 'Validation', 'Test']) 
    plt.show()   


def pred_vis(look_back, dates_train, dates_val, dates_test, y_train, y_val, y_test, 
             train_pred, val_pred, test_pred, show_train, show_val, show_test):

    if show_train:
        plt.figure(figsize=(12,5))
        plt.plot(dates_train[look_back:], train_pred)
        plt.plot(dates_train[look_back:], y_train)
        plt.legend(['Training Predictions','Training Observations'])
        plt.show()

    if show_val:
        plt.figure(figsize=(12,5))
        plt.plot(dates_val[look_back:], val_pred)
        plt.plot(dates_val[look_back:], y_val)
        plt.legend(['Validation Predictions','Validation Observations'])
        plt.show()

    if show_test:
        plt.figure(figsize=(12,5))
        plt.plot(dates_test[look_back:], test_pred)
        plt.plot(dates_test[look_back:], y_test)
        plt.legend(['Test Predictions','Test Observations'])
        plt.show()



def backtest_vis(df):
    display(df)
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['Long Cum Ret'])
    plt.plot(df.index, df['Strat Cum Ret'])
    plt.legend(['Long Returns','Strategy Returns'])
    plt.show()