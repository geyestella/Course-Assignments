# -*- coding: utf-8 -*-
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
print('train_images original shape is',train_images.shape)
'''
image = train_images[0].reshape([28,28])
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
'''
train_images_divide49=np.zeros((60000,49))
test_images_divide49=np.zeros((10000,49))
for time in range(0,60000):
  i=0
  j=0
  for n in range(0,49):
    train_images_divide49[time][n]=(train_images[time][i+0][j+0]+train_images[time][i+0][j+1]+
           train_images[time][i+0][j+2]+train_images[time][i+0][j+3]+train_images[time][i+1][j+0]
           +train_images[time][i+1][j+1]+train_images[time][i+1][j+2]+train_images[time][i+1][j+3]
           +train_images[time][i+2][j+0]+train_images[time][i+2][j+1]+train_images[time][i+2][j+2]
           +train_images[time][i+2][j+3]+train_images[time][i+3][j+0]+train_images[time][i+3][j+1]
           +train_images[time][i+3][j+2]+train_images[time][i+3][j+3])/16
    j=j+4
    if ((n+1)%7==0):
        i=i+4
        j=0
train_images_divide49=np.reshape(train_images_divide49,(60000,7,7))
for time in range(0,10000):
  i=0
  j=0
  for n in range(0,49):
    test_images_divide49[time][n]=(test_images[time][i+0][j+0]+test_images[time][i+0][j+1]+
           test_images[time][i+0][j+2]+test_images[time][i+0][j+3]+test_images[time][i+1][j+0]
           +test_images[time][i+1][j+1]+test_images[time][i+1][j+2]+test_images[time][i+1][j+3]
           +test_images[time][i+2][j+0]+test_images[time][i+2][j+1]+test_images[time][i+2][j+2]
           +test_images[time][i+2][j+3]+test_images[time][i+3][j+0]+test_images[time][i+3][j+1]
           +test_images[time][i+3][j+2]+test_images[time][i+3][j+3])/16
    j=j+4
    if ((n+1)%7==0):
        i=i+4
        j=0
test_images_divide49=np.reshape(test_images_divide49,(10000,7,7))
'''
image = np.array(train_images_divide49[0]).reshape([7,7])
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
'''


print('train_images shape after the trasition is', train_images_divide49.shape)
print('test_images shape after the trasition is', test_images_divide49.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,7)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
start = datetime.datetime.now()
model.fit(train_images_divide49, train_labels, epochs=5)
end = datetime.datetime.now()
test_loss, test_acc = model.evaluate(test_images_divide49, test_labels)

print('Test accuracy:', test_acc)
print ('The training time is:', end-start)