# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:56:52 2020

@author: Navdeep Thakur
"""

# Importing Necessary Libraries for Time Series Forecast Model
import pandas as pd
import numpy as np
import seaborn as sns
from fbprophet import Prophet
import plotly.express as px
import plotly.io as pio

pio.renderers.default='browser'

# Data Ingestion

invdat = pd.read_excel('Usecase2_Dataset.xlsx')


# Feature Engineering and Data Pre-processing

invdat_unpivoted = invdat.melt(id_vars=['Part No'], var_name='Wk_Month_Yr', value_name='Qty')

invdat_unpivoted.info()

# Removing Blank & White Spaces

invdat_unpivoted_ws = invdat_unpivoted.replace(r'^\s*$', np.nan, regex=True)

invdat_unpivoted_ws.columns = invdat_unpivoted_ws.columns.str.replace(' ', '')

invdat_unpivoted_ws.info()

# Creating Week, Month, and Year Columns

invdat_unpivoted_ws[['Week','Month','Year']] = invdat_unpivoted_ws.Wk_Month_Yr.str.split("-",expand=True)

# Checking no of different parts
len(invdat_unpivoted_ws['PartNo'].unique())

# Creating test and train datasets
test_data = invdat_unpivoted_ws[invdat_unpivoted_ws['Year']=='2019']

train_data = invdat_unpivoted_ws.drop(invdat_unpivoted_ws[invdat_unpivoted_ws['Year'] == '2019' ].index)

train_data.info()


# Creating Month and Year Column

data = train_data

data['M_Y'] = data['Month'] + "-" + data['Year']


data01 = data.loc[(train_data['PartNo']==29032636) & (train_data['Year']=='2018')]
#data = train_data.loc[train_data['PartNo']==29032636]


# EDA
sns.factorplot(data = data01, x ="Wk_Month_Yr", y = "Qty",col = 'PartNo')

fig = px.line(data01, x='Wk_Month_Yr', y='Qty')
fig.show()


# Creating date time column from Mont Year column
data01.loc[:,'M_Y'] =  pd.to_datetime(data01.loc[:,'M_Y'], format='%b-%Y').dt.date

import datetime

# Creating Week Start dates

for i in range(1, len(data01['M_Y'])):
    data01.iloc[i,6] = data01.iloc[i-1,6] + datetime.timedelta(days=7)
    


data01.columns = data.columns.str.replace(' ', '')
    
data01.info()

# Prophet Model

data01_model = Prophet(interval_width=0.95)

data01 = data01.rename(columns={'M_Y': 'ds', 'Qty': 'y'})

data01_model.fit(data01)

data_forecast = data01_model.make_future_dataframe(periods=12, freq='W')
data_forecast_pred = data01_model.predict(data_forecast)

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
data01_model.plot(data_forecast_pred, xlabel = 'Date', ylabel = 'Qty')
plt.title('Qty Forecast')


#fig = px.line(data01, x='ds', y='y')
#fig.show()
#
#data.info()


data01['ds'] = pd.to_datetime(data01['ds'], format='%Y-%m-%d').dt.date

data01.info()



fdata = data01[['ds','y']]

fdata.set_index('ds', inplace=True)


fdata.plot()

fdata.info()

fdata.columns

# ARIMA Model

from pmdarima.arima import ADFTest

from pmdarima.arima import auto_arima

adf_test = ADFTest(alpha = 0.05)

adf_test.should_diff(fdata)

ftrain = fdata[:40]

ftest = fdata[40:]

plt.plot(ftrain)

plt.plot(ftest)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

arima_model =  auto_arima(ftrain,start_p=0, d=1, start_q=0, 
                          max_p=5, max_d=5, max_q=5, start_P=0, 
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True, 
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 50 )

arima_model.summary()

prediction = pd.DataFrame(arima_model.predict(n_periods = 12),index=ftest.index)

prediction.columns = ['predicted_qty']
prediction


plt.figure(figsize=(8,5))
plt.plot(ftrain,label="Training")
plt.plot(ftest,label="Test")
plt.plot(prediction,label="Predicted")
plt.legend(loc = 'Left corner')
plt.show()


# LSTM Model
train_lstm = ftrain
test_lstm = ftest

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_lstm)
scaled_train_data = scaler.transform(train_lstm)
scaled_test_data = scaler.transform(test_lstm)


from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 39
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator,epochs=45)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm)

lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_lstm)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)



lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions


test_lstm['LSTM_Predictions'] = lstm_predictions
test_lstm

test_lstm['y'].plot(figsize = (16,5), legend=True)
test_lstm['LSTM_Predictions'].plot(legend = True);

























