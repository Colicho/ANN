import os
import tensorflow as tf
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update(plt.rcParamsDefault)
# C:\Users\tilfr\Documents\A_TUM\Master\Masterthesis\GitLab\system-level-pas\stylelib
plt.style.use(os.path.expanduser('~') + r'\Documents\A_TUM\Master\Masterthesis\GitLab\system-level-pas\stylelib\scientific.mplstyle')
# plt.style.use(os.path.expanduser('~') + r'\Documents\A_TUM\Master\Masterthesis\GitLab\system-level-pas\stylelib\ppt.mplstyle')
cmap = mpl.cm.get_cmap("plasma")

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


# create train (validation) and test data
time_vec = np.linspace(0, 1600, 1601).astype(int)
x = mackey_glass_series(time=time_vec)

t_train = np.linspace(300, 1300, 1001).astype(int)
t_test = np.linspace(1301, 1500, 200).astype(int)

df_x_train = pd.DataFrame(columns=[-20, -15, -10, -5, 0], index=range(len(t_train)-1))
df_x_test = pd.DataFrame(columns=[-20, -15, -10, -5, 0], index=range(len(t_test)-1))
df_y_train = pd.DataFrame(columns=[5], index=range(len(t_train)-1))
df_y_test = pd.DataFrame(columns=[5], index=range(len(t_test)-1))

for i, dt in enumerate([20, 15, 10, 5, 0]):
    df_x_train.iloc[:, i] = x[t_train[0] - dt: t_train[-1] - dt]
    df_x_test.iloc[:, i] = x[t_test[0] - dt: t_test[-1] - dt]

df_y_train.iloc[:, 0] = x[t_train[0]+5: t_train[-1]+5]
df_y_test.iloc[:, 0] = x[t_test[0]+5: t_test[-1]+5]


#### plot mackey_glass function
fig1, ax = plt.subplots(1)
plt.plot(x)
ax.set_xlabel(r'time')
ax.set_ylabel(r'')
ax.set_xlim([0, 1500])
plt.show()


x_train_tens = tf.convert_to_tensor(np.array(df_x_train).astype('float32'))
y_train_tens = tf.convert_to_tensor(np.array(df_y_train).astype('float32'))
x_test_tens = tf.convert_to_tensor(np.array(df_x_test).astype('float32'))
y_test_tens = tf.convert_to_tensor(np.array(df_y_test).astype('float32'))

x_train_tens = tf.stack(x_train_tens)
y_train_tens = tf.stack(y_train_tens)
x_test_tens = tf.stack(x_test_tens)
y_test_tens = tf.stack(y_test_tens)



# parameters for Network architecture and training
n_h1 = [3, 3, 3, 4, 4, 4, 5, 5, 5]  # number of hidden neurons layer 1
n_h2 = 3*[2, 4, 6]  # number of hidden neurons layer 2
lambda_l2 = 0.001  # hyperparameter lambda for regularization
val_losses_vec = []
train_losses_vec = []
test_losses_vec = []

# loop over ten cycles (for statistics on architecture performance)
for j in range(10):
    models = []
    histories = []
    pred = []
    losses = []

    # loop over all combinations of n_h1 and n_h2
    for i in range(len(n_h2)):

        # Initialize Network Model
        models.append(Sequential([

            # input layer
            InputLayer(input_shape=(5,)),
            # Flatten(input_shape=(5, 1)),

            # dense layer 1
            Dense(n_h1[i], activation='sigmoid'),  # kernel_regularizer=l2(lambda_l2)),

            # dense layer 2
            Dense(n_h2[i], activation='sigmoid'),  # kernel_regularizer=l2(lambda_l2)),

            # output layer
            Dense(1, activation='linear')
        ])
        )

        models[i].compile(optimizer='adam', loss='mse', metrics=['mse'])

        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

        histories.append(models[i].fit(x_train_tens, y_train_tens, epochs=300,
                            batch_size=20,
                            validation_split=0.2,
                            verbose=0)
                            # ,callbacks=[es])
                          )

        # results = model.evaluate(x_test_tens, y_test_tens, verbose=0)
        loss, mae = models[i].evaluate(x_test_tens, y_test_tens)
        losses.append(loss)
        pred.append(models[i].predict(x_test_tens))
        print('valid error:', histories[i].history['val_loss'][-1])
        print('test loss, test acc:', loss, mae)
        # print('test loss, test acc:', results)

        # # plot Training and Validation error over number of epochs
        # fig2, ax = plt.subplots(1)
        # plt.plot(histories[i].history['loss'], label='train')
        # plt.plot(histories[i].history['val_loss'], label='val')
        # ax.set_xlabel(r'Epochs')
        # ax.set_ylabel(r'')
        # # ax.grid(True, which="both")
        # # plt.ylim(y_limit)
        # ax.legend(loc='upper right',
        #           bbox_to_anchor=(1, 0.5))
        # # ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
        # plt.show()
        #
        # # plot prediction of the test data by trained network and test data
        # fig3, ax = plt.subplots(1)
        # plt.plot(y_test_tens, label='test_data')
        # plt.plot(pred[i], label='prediction')
        # ax.set_xlabel(r'time')
        # ax.set_ylabel(r'')
        # # ax.grid(True, which="both")
        # # plt.ylim(y_limit)
        # ax.legend(loc='center left',
        #           bbox_to_anchor=(1, 0.5))
        # # ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
        # plt.show()
        #
        #
        #
        # SAVE_PLOT = 0
        # if SAVE_PLOT:
        #     # create figures folder
        #     FIGPATH = os.path.join(os.path.expanduser('~') + '\Documents\A_TUM\Master\KTH\ANNaDA', r'Figures')
        #     print(FIGPATH)
        #     if not os.path.exists(FIGPATH):
        #         os.mkdir(FIGPATH)
        # # for formatting in ['svg', 'png']:
        # #     name = f'M_G_function_{time.strftime("%Y%m%d-%H%M%S")}.' + formatting
        # #     figname = os.path.join(FIGPATH, name)
        # #     fig1.savefig(figname, format=formatting, bbox_inches='tight')
        # for formatting in ['svg', 'png']:
        #     name = f'Train_err_and_Val_err_vs_epochs_nh1_{n_h1[i]}_nh2_ES_{n_h2[i]}_{time.strftime("%Y%m%d-%H%M%S")}.' + formatting
        #     figname = os.path.join(FIGPATH, name)
        #     fig2.savefig(figname, format=formatting, bbox_inches='tight')
        # for formatting in ['svg', 'png']:
        #     name = f'Predicted_fct_vs_time_nh1_{n_h1[i]}_nh2_{n_h2[i]}_ES_{time.strftime("%Y%m%d-%H%M%S")}.' + formatting
        #     figname = os.path.join(FIGPATH, name)
        #     fig3.savefig(figname, format=formatting, bbox_inches='tight')



    # fig4, ax = plt.subplots(1)
    # plt.plot(y_test_tens, label='test_data')
    # # for i in range(2):
    # for i in range(len(n_h2)):
    #     plt.plot(pred[i], label=f'{n_h1[i]}, {n_h2[i]}', color=cmap(i/len(n_h2)))
    # ax.set_xlabel(r'time')
    # ax.set_ylabel(r'')
    # # ax.grid(True, which="both")
    # # plt.ylim(y_limit)
    # ax.legend(loc='center left',
    #           bbox_to_anchor=(1, 0.5))
    # # ax.get_legend().set_title(f'Dry Air (solid), Water vapor (dashed)')
    # plt.show()

    # save validation, training, and test errors for all n_h1/2 combinations and for each run
    val_losses = []
    train_losses = []
    test_losses = []
    for i in range(len(n_h2)):
        # print(histories[i].history['val_loss'][-1])
        val_losses.append(histories[i].history['val_loss'][-1])
        train_losses.append(histories[i].history['loss'][-1])

    val_losses_vec.append(val_losses)
    train_losses_vec.append(train_losses)
    test_losses_vec.append(losses)



# Boxplot for 10 runs for each network n_h1/2 combination
# data = [[v[i] for v in val_losses_vec] for i in range(len(n_h2))]  # sort data of validation errors for boxplot
# data = [[v[i] for v in train_losses_vec] for i in range(len(n_h2))]  # sort data of training errors for boxplot
data = [[v[i] for v in test_losses_vec] for i in range(len(n_h2))]  # sort data of test errors for boxplot

fig5, ax = plt.subplots(1)
bp = ax.boxplot(data)
ax.set_xticklabels([f'{n_h1[i]},{n_h2[i]}' for i in range(len(n_h2))])
ax.set_xlabel(r'Combinations of n_h1, n_h2')
ax.set_ylabel(r'Test error after training')
# show plot
# ax.set_ylim([0, 0.03])
plt.show()






SAVE_PLOT = 1
if SAVE_PLOT:
    # create figures folder
    FIGPATH = os.path.join(os.path.expanduser('~') + '\Documents\A_TUM\Master\KTH\ANNaDA', r'Figures')
    print(FIGPATH)
    if not os.path.exists(FIGPATH):
        os.mkdir(FIGPATH)
# for formatting in ['svg', 'png']:
#     name = f'All_Predicted_fcts_vs_time_ES_{time.strftime("%Y%m%d-%H%M%S")}.' + formatting
#     figname = os.path.join(FIGPATH, name)
#     fig4.savefig(figname, format=formatting, bbox_inches='tight')
for formatting in ['svg', 'png']:
    name = f'Boxplot_test_error_all_nh_combos_ES_{time.strftime("%Y%m%d-%H%M%S")}.' + formatting
    figname = os.path.join(FIGPATH, name)
    fig5.savefig(figname, format=formatting, bbox_inches='tight')
