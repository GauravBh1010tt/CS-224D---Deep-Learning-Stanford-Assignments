import gzip
from six.moves import cPickle
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils

f=gzip.open('mnist.pkl.gz','rb')
data = cPickle.load(f)
(X_train, y_train), (X_test, y_test) = data

'''x_train = np.zeros([60000,28*28])
x_test = np.zeros([10000,28*28])

for i in range(X_train.shape[0]):
    x_train[i,] = X_train[i,].reshape(28*28)

for i in range(X_test.shape[0]):
    x_test[i,] = X_test[i,].reshape(28*28)'''

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


model = Sequential()
model.add(Dense(64, input_dim=784, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=16)
score = model.evaluate(X_test, Y_test, batch_size=16)
