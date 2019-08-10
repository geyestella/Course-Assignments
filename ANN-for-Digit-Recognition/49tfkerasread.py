import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import adam
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = np.genfromtxt('train_7times7.csv', dtype=float, delimiter=',')
X_test = np.genfromtxt('test_7times7.csv', dtype=float, delimiter=',')

X_train=np.reshape(X_train,(60000,7,7))
X_test=np.reshape(X_test,(10000,7,7))
'''
print('train_images original shape is',train_images.shape)
image = train_images[0].reshape([28,28])
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
image = np.array(X_train[0]).reshape([7,7])
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
print('train_images shape after the trasition is', X_train.shape)
print('test_images shape after the trasition is', X_test.shape)
'''
# Set Model
model = Sequential()
model.add(Flatten(input_shape=(7,7)))
model.add(Dense(100, activation='relu'))
model.add(Dense(28, activation='softmax'))
# Set Optimizer
opt = adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
start = datetime.datetime.now()
model.fit(X_train, train_labels, epochs=5)
end = datetime.datetime.now()
test_loss, test_acc = model.evaluate(X_test, test_labels)
print('Test accuracy:', test_acc)
print ('The training time is:', end-start)