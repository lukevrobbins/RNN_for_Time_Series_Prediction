#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime 
from pandas import Series 
            
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)

plt.style.use('fivethirtyeight')


# # Data Importing and Exploration

# ## Importing data for ASIANPAINT company stock statistics

# In[2]:


url = 'https://drive.google.com/file/d/1nM8Lc7El4dr9eq-iMkhiw1z3ZnCQOejv/view?usp=sharing'
# path to data
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]


# In[3]:


df = pd.read_csv(path)
df


# ## Performing Exploratory Analysis

# In[4]:


df.describe()


# Things to note
# - There is a large range for most variables, considering this is only one stock ticker. This is due to the large time range.
# - The mean of most of the price metrics is around 1250 and SD is around 1075, so the stock price must have stayed around this price for a while. 
# - We need to look for missing values in these columns

# In[5]:


df.isnull().sum()


# We can see alot of missing values for "Trades" column and some for the deliverable columns. Likely trades will be a significant predictor for our model so we will need to test if deleting the rows with no trade value will affect test error. Deliverables will likely have no affect on prediction accuracy.
# 
# We can create a dataframe with NAs and without NAs for seperate testing to deal with these values. However, it will be hard use common substitutes for NA values as the range of trades is extremely large. This means we will need another substitute. 
# 

# In[6]:


df.dtypes


# We can see that our date column is an object type instead of a datetime. We will need to handle this in order to do time-series analysis. 
# As of now, we are just working with the ASIANPAINT stock, so we may be able to drop this column. This is the same for the series column, since all of the stocks we will be looking at are EQ series. 
# 
# 

# In[7]:


df = df.drop(columns = "Series")
df = df.drop(columns = "Symbol")
df = df.drop(columns = "%Deliverble")
df


# In[8]:


df.Date = pd.to_datetime(df.Date)
saturdays = df[df.Date.dt.dayofweek == 5]  # 20 saturday values
sundays = df[df.Date.dt.dayofweek == 6]  # 4 sunday values

# Deleting weekend values from df
weekends = pd.concat([saturdays, sundays])
df = pd.concat([df, weekends]).drop_duplicates(keep=False)
df


# In[9]:


df_vwap = pd.DataFrame()

df_vwap['year'] = df.Date.dt.year
df_vwap['date'] = df.Date
df_vwap['close'] = df.Close
df_vwap['day of week'] = df.Date.dt.dayofweek

df_vwap.set_index('date', inplace = True)

plt.plot(df_vwap['close'])
plt.title("Closing Price Over Time")
plt.xlabel("Time in Years")
plt.ylabel("Close")


# In[10]:


df_vwap.groupby('year')['close'].mean().plot.bar()


# In[11]:


df_vwap.groupby('day of week')['close'].mean().plot.bar()


# In[12]:


# Showing rolling 30 day average
df_vwap['close'].plot()
df_vwap.rolling(window=30).mean()['close'].plot(figsize=(16, 6))


# In[13]:


# % of total Trades that are null
print(sum(df.Trades.isna()) / len(df.Trades))  # roughly 50% of Trades are null

# % of Deliverable Volume that is null 
print(sum(df['Deliverable Volume'].isna()) / len(df['Deliverable Volume']))


# In[14]:


# Replacing NA values with mean
df['Deliverable Volume'].fillna((df['Deliverable Volume'].mean()), inplace=True)
df['Deliverable Volume'] = round(df['Deliverable Volume'], 2)

# Dropping Trades column as there is no sufficient way to replace NAs and we have volume and VWAP to use instead
df = df.drop(columns = "Trades")
df.set_index('Date', inplace = True)

df.describe()


# In[15]:


# add a prediction column for close prices of next day
# Shift Close up one row, append current (row i+1) value to pred_c column
df['pred_c'] = df.Close.shift(-1)

# drops last row, which is missing pred_c data
df = df.dropna()
df


# In[16]:


values = df.values
values = values.astype('float32')
values


# In[17]:


scaler = MinMaxScaler(feature_range= (0,1))

scaled = scaler.fit_transform(values)
scaled = pd.DataFrame(scaled)
scaled.columns = df.columns
scaled['Date'] = df.index
scaled.set_index('Date', inplace = True)
scaled


# In[18]:


# planning out shape of data

# shape of input:
# (number_of_records x length_of_sequence x types_of_sequences)
# (5305 * 1 x 10)

# shape of output:
# (number_of_records x types_of_sequences)
# (5305 * 1)

scaled


# In[19]:


# Organizing training and testing data
X = scaled.drop(columns=['pred_c']).to_numpy()
X = np.expand_dims(X, axis=2)

Y = scaled.pred_c.to_numpy() 
Y = np.expand_dims(Y, axis=1)

X.shape, Y.shape

# Split 80/20
cutoff = int(X.shape[0] * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
Y_train, Y_test = Y[:cutoff], Y[cutoff:]
print(X.shape, Y.shape)


# In[20]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[21]:


def relu(x):
    return max(0.0, np.average(x))


# In[22]:


def forward_pass(x, seq_len, prev_act, U, V, W, verbose=False):
    layers = []
    i = 0
    while i < seq_len:
        input = np.zeros(x.shape)    
        input[i] = x[i]              
        new_u = np.dot(U, input)
        new_w = np.dot(W, prev_act)
        add = new_w + new_u
        act = relu(add)
        new_v = np.dot(V, act)
        prev_act = act 
        layers.append({'act': act, 'prev_act': prev_act})
        i += 1
    weights = {'new_u': new_u, 'new_v': new_v, 'new_w': new_w}
    if verbose:
        return (layers, add, weights)
    else:
        return weights 


# In[23]:


def get_loss(X, Y, seq_len, U, V, W):
    loss = 0.0
    for x, y in zip(X, Y):
        prev_act = np.zeros((hidden_layers, 1))   
        weights = forward_pass(x, seq_len, prev_act, U, V, W)
        loss += (y - weights['new_v'])**2 / 2
    
    return loss / float(y.shape[0])


# In[24]:


def train_model(X, Y, seq_len, learning_rate, U, V, W):
    for x, y in zip(X, Y):
        d_U = d_U_t = np.zeros(U.shape)
        d_V = d_V_t = np.zeros(V.shape)
        d_W = d_W_t = np.zeros(W.shape)
        
        # Forward pass
        prev_act = np.zeros((hidden_layers, 1))
        layers, add, weights = forward_pass(x, seq_len, prev_act, U, V, W, verbose=True)
        d_new_v = (weights['new_v'] - y)
        
        # Backward pass
        i = 0
        while i < seq_len:
            d_act = d_base = np.dot(np.transpose(V), d_new_v)
            d_add = add * (1 - add) * d_act

            d_new_w = d_add * np.ones_like(weights['new_w'])
            d_prev_act = np.dot(np.transpose(W), d_new_w)

            d_V_t = np.dot(d_new_v, np.transpose(layers[i]['act']))

            for _ in range(i-1, -1, -1):
                d_act = d_base + d_prev_act
                d_add = add * (1 - add) * d_act

                d_new_w = np.ones_like(weights['new_w']) * d_add  
                d_new_u = np.ones_like(weights['new_u']) * d_add 
                d_prev_act = np.dot(np.transpose(W), d_new_w)

                input = np.zeros(x.shape)
                input[i] = x[i]

                d_U_t += np.dot(U, input) 
                d_W_t += np.dot(W, layers[i]['prev_act'])

            # Update weights
            d_V += d_V_t
            d_U += d_U_t
            d_W += d_W_t

            # Limit weights
            for d in [d_U, d_V, d_W]:
                if d.max() > max_d:
                    d[d > max_d] = max_d
                if d.min() < min_d:
                    d[d < min_d] = min_d

            i += 1

        U -= learning_rate * d_U
        V -= learning_rate * d_V
        W -= learning_rate * d_W

    return (U, V, W)


# In[25]:


def predict(X, Y, seq_len, weights, hidden_layers):
    U, V, W = weights 
    pred = []
    for x, y in zip(X, Y):
        prev_act = np.zeros((hidden_layers, 1))
        i = 0
        while i < seq_len:
            new_u = np.dot(U, x)
            new_w = np.dot(W, prev_act)
            add = new_w + new_u
            act = relu(add)
            new_v = np.dot(V, act)
            prev_act = act
            i += 1
        pred.append(new_v)
    return np.array(pred)


# In[26]:


def accuracy(actual, predicted):
    i = 0
    correct = 0
    while i < len(actual):
        if abs(actual[i] - predicted[i]) < 0.01:
            correct += 1
        i += 1
    return correct / float(len(actual)) * 100.0


# In[27]:


# hyperparameters to tune: epochs, learning rate, hidden_layersrs
t_epochs = [5, 7, 15]
t_learning_rates = [0.0001, .001, .01]
t_hidden_layers = [1, 2]


min_d = -10
max_d = 10
output_dim = 1
seq_len = 10                   


# In[28]:


e_list = []
l_list = []
h_list = []
train_rmse_list = []
test_rmse_list = []
train_acc_list = []
test_acc_list = []


# In[29]:


import timeit
from sklearn.metrics import mean_squared_error
import math

start = timeit.default_timer()
final_output = pd.DataFrame()
output = pd.DataFrame()

for epochs in t_epochs:
    for learning_rate in t_learning_rates:
        for hidden_layers in t_hidden_layers:
            # train
            U = np.random.uniform(0, 1, (hidden_layers, seq_len))
            W = np.random.uniform(0, 1, (hidden_layers, hidden_layers))
            V = np.random.uniform(0, 1, (output_dim, hidden_layers))

            for epoch in range(epochs):
                # check loss on train
                loss = get_loss(X_train, Y_train, seq_len, U, V, W)
                val_loss = get_loss(X_test, Y_test, seq_len, U, V, W)
            
                print('Epoch: ', epoch + 1, '| Loss:', loss[0][0], '| Val Loss:', val_loss[0][0])
            
                U, V, W = train_model(X_train, Y_train, seq_len, learning_rate, U, V, W)
            
            # get predictions
            train_pred = predict(X_train, Y_train, seq_len, (U, V, W), hidden_layers)
            test_pred = predict(X_test, Y_test, seq_len, (U, V, W), hidden_layers)
            
            # plot predictions vs actual
            plt.plot(train_pred[:, 0, 0], 'g')
            plt.plot(Y_train[:, 0], 'c')
            plt.show()
            
            plt.plot(test_pred[:, 0, 0], 'g')
            plt.plot(Y_test[:, 0], 'c')
            plt.show()
            
            # get RMSE
            train_rmse = math.sqrt(mean_squared_error(Y_train[:, 0], train_pred[:, 0, 0]))
            test_rmse = math.sqrt(mean_squared_error(Y_test[:, 0], test_pred[:, 0, 0]))
            
            print('Epochs: ', epochs)
            print('Learning Rate: ', learning_rate)
            print('Hidden Layers: ', hidden_layers)
            print('Training RMSE:', train_rmse)
            print('Testing RMSE', test_rmse)
            
            # get accuracy
            train_accuracy = round(accuracy(Y_train[:, 0], train_pred[:, 0, 0]), 2)
            test_accuracy = round(accuracy(Y_test[:, 0], test_pred[:, 0, 0]), 2)

            print(f'Training accuracy: {train_accuracy}%')
            print(f'Testing accuracy: {test_accuracy}%')

            # output to dataframe
            
            e_list.append(epochs)
            l_list.append(learning_rate)
            h_list.append(hidden_layers)
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            
            
end = timeit.default_timer()

print(f'time: {end-start} s')

d = {'Epochs' : e_list, 'Learning Rate' : l_list, 'Hidden Layers' : h_list, 'Train RMSE' : train_rmse_list,
    'Test RMSE' : test_rmse_list, 'Training Accuracy' : train_acc_list, 'Test Accuracy' : test_acc_list}
deedfR = pd.DataFrame(d)


# In[30]:


deedfR

