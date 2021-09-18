# -*- coding: utf-8 -*-
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

(trainX, trainY), (testX, testY) = mnist.load_data()

#reshape
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

#onehot encoding
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)

#normalising pixels
trainX = np.array(trainX, dtype='float64')
testX = np.array(testX, dtype='float64')
trainX/=255
testX/=255

#I used this mode in a previous assignment in University
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

#fitting the model and then plotting it
history = model.fit(trainX, trainY, batch_size=32, epochs=10, validation_split=0.1)
plt.figure()
plt.plot(history.history['loss'])
plt.grid()

#predictions
predY = np.argmax(model.predict(testX), axis=-1)

#metrics calculated here
cm = confusion_matrix(np.argmax(testY, axis=1), predY)
acc = 100*accuracy_score(testY, to_categorical(predY, num_classes=10))