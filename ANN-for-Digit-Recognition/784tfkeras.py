# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import datetime
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
start = datetime.datetime.now()
model.fit(train_images, train_labels, epochs=5)
end = datetime.datetime.now()
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
print ('The training time is:', end-start)