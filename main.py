import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

from random import randint
from itertools import cycle

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
def mlp(train_data, valid_data, n_classes=10, n_inner_layers=2, units_per_layer=(512, 512),
        activations=('relu', 'relu', 'softmax'), dropouts=None,
        optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
        metrics=None, batch_size=64, epochs=10):
    """
    Create a mlp estimator using keras library, train this model and return it.

    Args:
        train_data (tuple): Tuple containing inputs and labels as well (x, y).
        valid_data (tuple): Same as train_data, this data is used to validate.
        n_classes (int): Number of classes.
        n_inner_layers (int): Number of inner layers.
        units_per_layer (list): List of amounts of units per layer, the length of this list has to be the same
        as the number of inner layers.
        activations (list): List of activation functions, the length of this list is 1 element larger than
        units_per_layer.
        dropouts (list): List of dropout rates, the length of this list gas to be the same as the number
        of inner layers.
        optimizer (:obj:): Keras optimizer.
        loss (:obj:): Keras loss function.
        metrics (list): List of metrics.
        batch_size (int): Size of the batches.
        epochs (int): Number of epochs to train.

    Returns:
        model_hist (:obj:): Keras estimator model.
    """
    x_tr, y_tr = train_data
    model = Sequential()
    model.add(InputLayer(input_shape=x_tr.shape[1:]))
    for layer in range(n_inner_layers):
        model.add(Dense(units=units_per_layer[layer], activation=activations[layer]))
        if dropouts:
            model.add(Dropout(rate=dropouts[layer]))
    # Since with two classes we only need 1 decider neuron
    if n_classes == 2:
        n_classes -= 1
    model.add(Dense(units=n_classes, activation=activations[-1]))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(x=x_tr, y=y_tr, batch_size=batch_size, epochs=epochs, validation_data=valid_data)

    return model


# %%
mlp_model = mlp(train_data=(mlp_x_train, y_train), valid_data=(mlp_x_valid, y_valid), n_inner_layers=1,
                units_per_layer=[5], activations=['sigmoid', 'softmax'],
                optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

# %%
mlp_y_scores = mlp_model.predict(mlp_x_test)

# %%
mlp_y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# %%
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_digits):
    fpr[i], tpr[i], _ = roc_curve(mlp_y_test[:, i], mlp_y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# %%
fpr["micro"], tpr["micro"], _ = roc_curve(mlp_y_test.ravel(), mlp_y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# %%
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_digits)]))
lw = 2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_digits):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_digits

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_digits), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

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
