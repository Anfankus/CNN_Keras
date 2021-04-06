"""
@cite Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

"""
  VGG16/19 implemention according to 'Very Deep Convolutional Networks for Large-Scale Image Recognition'

  input_shape : the image shape, default is (299,299,3)
"""

def VGG16(input_shape, num_classes):
  model = k.models.Sequential(
    [
      Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal", input_shape=input_shape),
      Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Flatten(),
      Dense(4096, activation="relu",kernel_initializer="normal"),
      Dense(4096, activation="relu",kernel_initializer="normal"),
      Dense(num_classes, activation="softmax",kernel_initializer="normal"),
    ]
  )
  return model

def VGG19(input_shape, num_classes):
  model = k.models.Sequential(
    [
      Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal", input_shape=input_shape),
      Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="normal"),
      MaxPooling2D(pool_size=(2,2), strides=(2,2)),

      Flatten(),
      Dense(4096, activation="relu",kernel_initializer="normal"),
      Dense(4096, activation="relu",kernel_initializer="normal"),
      Dense(num_classes, activation="softmax",kernel_initializer="normal"),
    ]
  )
  return model
