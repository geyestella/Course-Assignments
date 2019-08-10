from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.neural_network import MLPClassifier 
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
read_start = datetime.datetime.now()
X_train = np.genfromtxt('train_7times7.csv', dtype=float, delimiter=',')
X_test = np.genfromtxt('test_7times7.csv', dtype=float, delimiter=',')
read_end = datetime.datetime.now()
mlp = MLPClassifier(hidden_layer_sizes=(100,100),learning_rate='constant',learning_rate_init=0.01,max_iter=20)
start = datetime.datetime.now()
mlp.fit(X_train, train_labels)
end = datetime.datetime.now()
# Returns a NumPy Array
# Predict for One Observation (image)
print("Training set score: %f" % mlp.score(X_train, train_labels))
print("Test set score: %f" % mlp.score(X_test, test_labels))
print ('The training time is:', end-start)
print ('The read time is:', read_end-read_start)





'''
X_train=np.reshape(X_train,(60000,7,7))
X_test=np.reshape(X_test,(10000,7,7))
'''
