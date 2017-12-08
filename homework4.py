import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers
from keras.utils import np_utils


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    ytrain_1hot = np_utils.to_categorical(ytrain)
    xtrain = xtrain/255

    xtest, ytest = test
    ytest_1hot = np_utils.to_categorical(ytest)
    xtest = xtest/255

    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    # After training:
    # >>> nn.evaluate(xtest,ytest_1hot)
    # 10000/10000 [==============================] - 1s 67us/step
    # [1.4281578912734985, 0.49330000000000002]

    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)
    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn


def train_multilayer_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=5, batch_size=32)


def build_convolution_nn():
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same",input_shape=(32,32,3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.5))
    nn.add(Flatten())
    nn.add(Dense(units=250, activation="relu"))
    nn.add(Dense(units=100, activation="relu"))
    nn.add(Dense(units=10, activation='softmax'))
    return nn


def train_convolution_nn(model, xtrain, ytrain, learning_rate, ep, bs):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=ep, batch_size=bs)


def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    ytrain_1hot = [1 if e>1 and e<8 else 0 for e in ytrain]
    xtrain = xtrain/255

    xtest, ytest = test
    ytest_1hot = np.array([1 if e>1 and e<8 else 0 for e in ytest])
    xtest = xtest/255

    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_binary_classifier():
    #>>> nn.evaluate(xt,yt)
    #10000/10000 [==============================] - 27s 3ms/step
    #[1.0367493948936461, 0.72489999999999999]
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same",input_shape=(32,32,3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    #nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    #nn.add(Dropout(0.5))
    nn.add(Flatten())
    nn.add(Dense(units=250, activation="relu"))
    nn.add(Dense(units=100, activation="relu"))
    nn.add(Dense(units=1, activation="sigmoid"))
    return nn


def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)


if __name__ == "__main__":

    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    nn = build_convolution_nn()
    train_convolution_nn(nn, xtrain, ytrain_1hot)
    nn.evaluate(xtest, ytest_1hot)
