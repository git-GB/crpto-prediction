##############################################################################
####################### IMPORTING ALL REQUIRED PACKAGES ######################
##############################################################################

#Importing all required packages
from pandas_datareader import data as pdr
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
yf.pdr_override()


##############################################################################
############### FUNCTIONS FOR QUICK DATA IMPORTING AND CLEANING ##############
##############################################################################

#A function to import data from yahoo finance-
def y_importer(y):
    start_date = '2018-06-01'
    end_date = '2021-06-01'
    x = pdr.get_data_yahoo(y, start=start_date, end=end_date)
    return x

# a function with a for loop to fill missing values with forward fill
#ie. the previous value
def fill_missing_val(x):
        for col in x:
            x[col].fillna(method='ffill',inplace = True)
        return   x.isnull().any()

#creating a new function to clean data
def crypto_cleaner(x):
    x['Change %']= x['Close'].pct_change()
    x['Change %'].fillna(method = 'bfill',inplace = True)
    x.reset_index(inplace=True,drop=False)
    x.Date = x.Date.astype('datetime64')
    return x.info()

#creating a new function to plot data
def crypto_plotter(x):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(x['Date'],x['Close'])
    ax2.plot(x['Date'],x['Change %'],color = 'g')
    ax3.plot(x['Date'],x['Volume'],color = 'r')
    fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.tight_layout()
    plt.show()

#creating a function to drop unnecessary data
def crypto_dropper(x):
    x['date']=x['Date']
    x['close']=x['Close']
    x['pct_chng']=x['Change %']
    x['vol']=x['Volume']
    x.drop(x.iloc[:,0:8], axis = 1, inplace=True)
    return x

#creating a function to combine previous 3 functions
def crypto_processor(x):
    fill_missing_val(x)
    crypto_cleaner(x)
    #crypto_plotter(x)
    crypto_dropper(x)
    return x

##############################################################################
###############IMPORTING AND CLEANING ALL THE REQUIRED DATA ##################
##############################################################################

#just 10 lines of code to import and clean all the necessary data
btc = y_importer('BTC-USD')
crypto_processor(btc)

eth = y_importer('ETH-USD')
crypto_processor(eth)

ada = y_importer('ADA-USD')
crypto_processor(ada)

doge = y_importer('DOGE-USD')
crypto_processor(doge)

xrp = y_importer('XRP-USD')
crypto_processor(xrp)


scaler =MinMaxScaler(feature_range=(0,1))

scaled_price = scaler.fit_transform(eth['close'].values.reshape(-1,1))

prediction_days = 365


x_train, y_train = [], []

for x in range(prediction_days, len(scaled_price)):
    x_train.append(scaled_price[x - prediction_days:x, 0])
    y_train.append(scaled_price[x, 0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#creating the Neural network

model = Sequential()
model.add(LSTM)
