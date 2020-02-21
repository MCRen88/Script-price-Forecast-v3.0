"""
@author MCRen88
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1st column of csv file is "date" which we don't need. And 3 footer lines can also be skipped.
dataset = pd.read_csv("data/energy.csv", usecols=[1], engine='python', skipfooter=3)

# fix random seed for reproducibility
seed = np.random.seed(11)

# normalize the dataset
data_range = (-1, 1)
scaler = MinMaxScaler(feature_range=data_range)        
scaler.fit(dataset)
# scaler can also de-normalize the dataset by scaler.inverse_transform(), useful for actual prediction
dataset_scaled = scaler.transform(dataset)

# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    i_range = len(data) - look_back - 1
    #print(i_range)
    for i in range(0, i_range):
        dataX.append(data[i:(i+look_back)])    # index can move down to len(dataset)-1
        dataY.append(data[i + look_back])      # Y is the item that skips look_back number of items
    
    return np.array(dataX), np.array(dataY)

# accounting 120 columns for lag as feature for the next prediction, I have tried 10 and 40 as well but 120 produces the changes gradually and accurately (not very significantly accurate)
look_back = 120
dataX, dataY = create_dataset(dataset_scaled, look_back=look_back)

print("X shape:", dataX.shape)
print("Y shape:", dataY.shape)
   
print("Xt-119    Xt-118   Xt-117   ...    Xt        Y")
print("---------------------------------------------")

for i in range(len(dataX)): 
    print('%.2f  \t %.2f  \t  %.2f    ...   %.2f    %.2f' % (dataX[i][0][0], dataX[i][1][0], dataX[i][2][0], dataX[i][59][0], dataY[i][0]))

# Reshape to (samples, timestep, features)
dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
print("X shape:", dataX.shape)

# splitting the date into train and test value
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
dataX, dataY = shuffle(dataX, dataY, random_state=4)

trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2, random_state=11)

# create and fit the LSTM network
from keras.layers import Dropout
from keras.layers import Flatten

batch_size = 32
timesteps = trainX.shape[1]
input_dim = trainX.shape[2]

model = Sequential()
model.add(LSTM(30, input_shape=(timesteps, input_dim)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics = ['mean_squared_error'])

model.summary()

# calculation og MSE and validation MSE parallely
training = model.fit(trainX, trainY, epochs=10, batch_size=batch_size, validation_data = (testX, testY), verbose=1)

from keras.models import load_model

model.save('./model/trained_model.h5')
print('------------Model saved!----------')

#load_model('./model/trained_model.h5')

# visualise training history
plt.plot(training.history['mean_squared_error'])
plt.plot(training.history['val_mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc="upper right")
plt.show() # Seems like epochs are not necessary there is a good convergence in early stage itself

from sklearn.metrics import mean_absolute_error
# make predictions
trainPredict = model.predict(trainX, batch_size)
testPredict = model.predict(testX, batch_size)   

MAE = mean_absolute_error(testY, testPredict)
print('MAE: ',MAE)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(testY, testPredict)
print('MAPE: ',MAPE)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# now i have to generate the sequence of prediction for the period of 1st January till 31st March
def generate_seq(model, seq_length, seed_series, future_days):
    """
    model        :  keras model
    seq_length   :  previous timestamps neded to forecast next timestamp
    seed_series  :  initial values to start prediction, ex: [1,2,3,4]
    future_days  :  number of days to forecast timestamps for
    """
    # scale seed series
    seed_series = np.reshape(seed_series, (-1, 1))
    seed_series_scaled = scaler.transform(seed_series)
    
    result = list()
    in_series = list(seed_series_scaled)
    # extrapolate time series
    for _ in range(future_days):
        in_series = in_series[-seq_length:]
        # predict next time stamp
        next_timestamp = model.predict(np.reshape(in_series, (1, timesteps, input_dim)), verbose=0)
        # append to input
        in_series.append(next_timestamp)# += ' ' + float()
        result.append(next_timestamp)
        result = [float(res) for res in result]
        
    # unscale result
    result_unscaled = np.array(result)
    result_unscaled = np.reshape(result_unscaled, (-1, 1))
    result_unscaled = scaler.inverse_transform(result_unscaled)
    
    return result_unscaled


input_series = pd.read_csv("data/energy_last_days.csv", usecols=[1])
Predicted_Price = generate_seq(model, timesteps, input_series, 24)

# storing the results (Predicted_Price) in dataframe
df1 = pd.DataFrame(data=Predicted_Price, columns  = ["Pred_Price"])
#df1.head()

# genearating Data Sequence
DateTime  = pd.date_range(start='12-14-2019', end='12-15-2019', freq='1H')

# storing it to a dataframe
df2 = pd.DataFrame(data=DateTime, columns = ['time'])
#df2.head(24)

# final dataframe combination of date and predicted price
Final_Prediction = pd.concat(objs = [df2,df1], axis = 1)
#Final_Prediction.head(24)

# wrting it back to csv
Final_Prediction.to_csv('./result_predicted/forecasted_price.csv', sep =',',index = False)

