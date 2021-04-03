"""
@cite Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.
"""

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def AlexNet(input_shape, num_classes):
  model = k.models.Sequential(
    [
      Conv2D(filters=96,kernel_size=(11,11),strides=(4,4), padding="same", activation="relu",kernel_initializer="uniform",input_shape=input_shape),
      MaxPooling2D(pool_size = (3,3), strides=(2,2)),
      Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same",activation="relu",kernel_initializer="uniform"),
      MaxPooling2D(pool_size=(3,3),strides=(2,2)),
      Conv2D(filters=192*2, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu",kernel_initializer="uniform"),
      Conv2D(filters=192*2, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu",kernel_initializer="uniform"),
      Conv2D(filters=128*2, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu",kernel_initializer="uniform"),
      MaxPooling2D(pool_size=(2,2), strides=(1,1)),
      Flatten(),
      Dense(4096, activation="relu"),
      Dropout(rate=0.5),
      Dense(4096, activation="relu"),
      Dropout(rate=0.5),
      Dense(num_classes, activation="softmax")
    ]
  )
  return model