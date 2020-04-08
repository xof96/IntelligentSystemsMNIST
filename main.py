import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve, auc
import matplotlib.pyplot as plt
from random import randint

# %%
# the data, split between train-valid and test sets
(x_train_valid, y_train_valid), (x_test, y_test) = mnist.load_data()
n_digits = 10

# %%
# data split between train (75%) and valid (25%)
# train_test_split does it like that by default
x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid)

# %%
# input image dimensions
n_train, img_rows, img_cols = x_train.shape
n_valid, _, _ = x_valid.shape
n_test, _, _ = x_test.shape

# %%
# Before reshaping images, we are going to save an image for visualization
rand_num = randint(0, 9999)
test_image = x_test[rand_num]
test_label = y_test[rand_num]

# %%
# Reshaping images to use multilayer perceptron as estimator
mlp_x_train = x_train.reshape(n_train, img_rows * img_cols)
mlp_x_valid = x_valid.reshape(n_valid, img_rows * img_cols)
mlp_x_test = x_test.reshape(n_test, img_rows * img_cols)

# %%
mlp = Sequential()
mlp.add(InputLayer(input_shape=mlp_x_train.shape[1:]))
mlp.add(Dense(units=512, activation='relu'))
mlp.add(Dropout(rate=0.2))
mlp.add(Dense(units=512, activation='relu'))
mlp.add(Dropout(rate=0.2))
mlp.add(Dense(units=n_digits, activation='softmax'))

# %%
mlp.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])

# %%
mlp_hist = mlp.fit(x=mlp_x_train, y=y_train, batch_size=64, epochs=10, validation_data=(mlp_x_valid, y_valid))

# %%
# Reshaping images to use convolutional neural network as estimator
if K.image_data_format() == 'channels_first':
    cnn_x_train = x_train.reshape(n_train, 1, img_rows, img_cols)
    cnn_x_valid = x_valid.reshape(n_valid, 1, img_rows, img_cols)
    cnn_x_test = x_test.reshape(n_test, 1, img_rows, img_cols)
else:
    cnn_x_train = x_train.reshape(n_train, img_rows, img_cols, 1)
    cnn_x_valid = x_valid.reshape(n_valid, img_rows, img_cols, 1)
    cnn_x_test = x_test.reshape(n_test, img_rows, img_cols, 1)

cnn_input_shape = x_train.shape[1:]

# %%
# Scaling data
# TODO: replace name of variables by the variables you want to scale
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

# %%
# Convolutional Neural Network
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=cnn_input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(rate=0.2))
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(rate=0.2))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(rate=0.5))
cnn.add(Dense(n_digits, activation='softmax'))

# %%
cnn.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])

# %%
cnn_hist = cnn.fit(x=cnn_x_train, y=y_train, batch_size=64, epochs=10, validation_data=(cnn_x_valid, y_valid))