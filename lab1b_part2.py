import tensorflow as t
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as pt

def mackey_glass_series(time, beta=0.2, gamma=0.1, n=10, tau=25):
    x = []
    x.append(1.5)
    for t in time:
        if t < tau:
            x_1 = x[t] - gamma * x[t]
        else:
            x_1 = x[t] + (beta * x[t - tau]) / (1 + (x[t - tau]) ** n) - gamma * x[t]
        x.append(x_1)
    return x


# create data

time_vec = np.linspace(0, 1600, 1601).astype(int)
x = mackey_glass_series(time=time_vec)

t_train = np.linspace(300, 1300, 1001).astype(int)
t_test = np.linspace(1301, 1500, 200).astype(int)

x_train = [x[t_train[0]-i: t_train[-1]-i] for i in [20, 15, 10, 5, 0]]

# x_train0 = [x[t_train[0]-i: t_train[-1]-i] for i in [20, 15, 10, 5, 0]]
df_x_train = pd.DataFrame(index=[-20, -15, -10, -5, 0], columns=range(len(t_train)-1))
df_x_test = pd.DataFrame(index=[-20, -15, -10, -5, 0], columns=range(len(t_test)-1))
df_y_train = pd.DataFrame(index=[5], columns=range(len(t_train)-1))
df_y_test = pd.DataFrame(index=[5], columns=range(len(t_test)-1))

for i, dt in enumerate([20, 15, 10, 5, 0]):
    df_x_train.iloc[i, :] = x[t_train[0] - dt: t_train[-1] - dt]
    df_x_test.iloc[i, :] = x[t_test[0] - dt: t_test[-1] - dt]


df_y_train.iloc[0, :] = x[t_train[0]+5: t_train[-1]+5]
df_y_test.iloc[0, :] = x[t_test[0]+5: t_test[-1]+5]



n_hidden = 5  # number of hidden neurons

model = Sequential([

    # input layer
    Flatten(input_shape=(5, 1)),

    # dense layer 1
    Dense(n_hidden, activation='sigmoid'),

    # dense layer 2
    Dense(n_hidden, activation='sigmoid'),

    # output layer
    Dense(1, activation='linear')
])


model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])


model.fit(df_x_train, df_y_train, epochs=10,
          batch_size=1000,
          validation_split=0.2)

results = model.evaluate(x_test, y_test, verbose = 0)
print('test loss, test acc:', results)



