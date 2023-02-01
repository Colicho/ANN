import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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



# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0



# create data

time_vec = np.linspace(0, 1600, 1601).astype(int)
x = mackey_glass_series(time=time_vec)

t_train = np.linspace(300, 1300, 1001).astype(int)
t_test = np.linspace(1301, 1500, 200).astype(int)

# x_train = [x[t_train[0]-i: t_train[-1]-i] for i in [20, 15, 10, 5, 0]]

# x_train0 = [x[t_train[0]-i: t_train[-1]-i] for i in [20, 15, 10, 5, 0]]
df_x_train = pd.DataFrame(columns=[-20, -15, -10, -5, 0], index=range(len(t_train)-1))
# df_x_train = pd.DataFrame(index=[-20, -15, -10, -5, 0], columns=range(len(t_train)-1))
df_x_test = pd.DataFrame(columns=[-20, -15, -10, -5, 0], index=range(len(t_test)-1))
# df_x_test = pd.DataFrame(index=[-20, -15, -10, -5, 0], columns=range(len(t_test)-1))
df_y_train = pd.DataFrame(columns=[5], index=range(len(t_train)-1))
# df_y_train = pd.DataFrame(index=[5], columns=range(len(t_train)-1))
df_y_test = pd.DataFrame(columns=[5], index=range(len(t_test)-1))
# df_y_test = pd.DataFrame(index=[5], columns=range(len(t_test)-1))


for i, dt in enumerate([20, 15, 10, 5, 0]):
    df_x_train.iloc[:, i] = x[t_train[0] - dt: t_train[-1] - dt]
    df_x_test.iloc[:, i] = x[t_test[0] - dt: t_test[-1] - dt]

df_y_train.iloc[:, 0] = x[t_train[0]+5: t_train[-1]+5]
df_y_test.iloc[:, 0] = x[t_test[0]+5: t_test[-1]+5]


#### plot mackey_glass function


fig1, ax = plt.subplots(1)
plt.plot(x, label='M-G function')
ax.set_xlabel(r'time')
ax.set_ylabel(r'')
# ax.grid(True, which="both")
# plt.ylim(y_limit)
ax.legend(loc='upper right')
          # bbox_to_anchor=(1, 0.5))
# ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
ax.set_xlim([0, 1500])
plt.show()



# for i, dt in enumerate([20, 15, 10, 5, 0]):
#     df_x_train.iloc[i, :] = x[t_train[0] - dt: t_train[-1] - dt]
#     df_x_test.iloc[i, :] = x[t_test[0] - dt: t_test[-1] - dt]
#
# df_y_train.iloc[0, :] = x[t_train[0]+5: t_train[-1]+5]
# df_y_test.iloc[0, :] = x[t_test[0]+5: t_test[-1]+5]


x_train_tens = tf.convert_to_tensor(np.array(df_x_train).astype('float32'))
y_train_tens = tf.convert_to_tensor(np.array(df_y_train).astype('float32'))
x_test_tens = tf.convert_to_tensor(np.array(df_x_test).astype('float32'))
y_test_tens = tf.convert_to_tensor(np.array(df_y_test).astype('float32'))

x_train_tens = tf.stack(x_train_tens)
y_train_tens = tf.stack(y_train_tens)
x_test_tens = tf.stack(x_test_tens)
y_test_tens = tf.stack(y_test_tens)


# arr_train = np.array(df_x_train).astype('float32')
#
# tensor_train = tf.convert_to_tensor(arr_train)



n_h1 = [3, 4, 5]  # number of hidden neurons
n_h2 = [2, 4, 6]  # number of hidden neurons

for i in range(1):

    model = Sequential([

        # input layer
        InputLayer(input_shape=(5,)),
        # Flatten(input_shape=(5, 1)),

        # dense layer 1
        Dense(n_h1[i], activation='sigmoid'),

        # dense layer 2
        Dense(n_h2[i], activation='sigmoid'),

        # output layer
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    history = model.fit(x_train_tens, y_train_tens, epochs=300,
                        batch_size=20,
                        validation_split=0.2,
                        verbose=0)

    # results = model.evaluate(x_test_tens, y_test_tens, verbose=0)
    loss, mae = model.evaluate(x_test_tens, y_test_tens)
    print('test loss, test acc:', loss, mae)
    # print('test loss, test acc:', results)


    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='val')
    #
    # plt.show()


    fig1, ax = plt.subplots(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'')
    # ax.grid(True, which="both")
    # plt.ylim(y_limit)
    ax.legend(loc='upper right')
              # bbox_to_anchor=(1, 0.5))
    # ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
    plt.show()


    pred = model.predict(x_test_tens)

    fig2, ax = plt.subplots(1)
    plt.plot(pred, label='prediction')
    plt.plot(y_test_tens, label='test_data')
    ax.set_xlabel(r'time')
    ax.set_ylabel(r'')
    # ax.grid(True, which="both")
    # plt.ylim(y_limit)
    ax.legend(loc='upper right')
              # bbox_to_anchor=(1, 0.5))
    # ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
    plt.show()
