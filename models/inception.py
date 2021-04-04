"""
@cite Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.
"""

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D

class InceptionV1Layer(k.Model):
  def __init__(self, x1, reduce_x3, x3, reduce_x5, x5, pool_proj):
    super(InceptionV1Layer,self).__init__()
    self.conv1 = Conv2D(x1, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")

    self.conv2_1 = Conv2D(reduce_x3, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")
    self.conv2_2 = Conv2D(x3, (3,3), (1,1), "same", activation="relu", kernel_initializer="normal")

    self.conv3_1 = Conv2D(reduce_x5, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")
    self.conv3_2 = Conv2D(x5, (5,5), (1,1), "same", activation="relu", kernel_initializer="normal")

    self.pool4_1 = MaxPooling2D((3,3),(1,1),"same")
    self.pool4_2 = Conv2D(pool_proj,(1,1),(1,1), "same", activation="relu", kernel_initializer="normal")
    
  def call(self, x):
    conv_path1 = self.conv1(x)
    conv_path2 = self.conv2_2(self.conv2_1(x))
    conv_path3 = self.conv3_2(self.conv3_1(x))
    conv_path4 = self.pool4_2(self.pool4_1(x))
    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])


def InceptionV1(input_shape, num_classes):
  inputs = k.Input(shape=input_shape)
  x = Conv2D(filters=64, kernel_size=7, strides=(2,2), activation="relu", padding="same", kernel_initializer="normal")(inputs)
  x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)
  x = Conv2D(filters=192, kernel_size=3, strides=1, activation="relu",  padding="same", kernel_initializer="normal")(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Layer(x1=64, reduce_x3=96, x3=128, reduce_x5=16, x5=32, pool_proj=32)(x)
  x = InceptionV1Layer(x1=128, reduce_x3=128, x3=192, reduce_x5=32, x5=96, pool_proj=64)(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Layer(x1=192, reduce_x3=96, x3=208, reduce_x5=16, x5=48, pool_proj=64)(x)

  auxiliary1 = AveragePooling2D(pool_size=5, strides=3)(x)
  auxiliary1 = Conv2D(128, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")(auxiliary1)
  auxiliary1 = Flatten()(auxiliary1)
  auxiliary1 = Dense(units=1024, activation="relu")(auxiliary1)
  auxiliary1 = Dropout(rate = 0.7)(auxiliary1)
  auxiliary1 = Dense(units = num_classes, activation="softmax")(auxiliary1)

  x = InceptionV1Layer(x1=160, reduce_x3=112, x3=224, reduce_x5=24, x5=64, pool_proj=64)(x)
  x = InceptionV1Layer(x1=128, reduce_x3=128, x3=256, reduce_x5=24, x5=24, pool_proj=64)(x)
  x = InceptionV1Layer(x1=112, reduce_x3=144, x3=288, reduce_x5=32, x5=64, pool_proj=64)(x)

  auxiliary2 = AveragePooling2D(pool_size=5, strides=3)(x)
  auxiliary2 = Conv2D(128, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")(auxiliary2)
  auxiliary2 = Flatten()(auxiliary2)
  auxiliary2 = Dense(units=1024, activation="relu")(auxiliary2)
  auxiliary2 = Dropout(rate = 0.7)(auxiliary2)
  auxiliary2 = Dense(units = num_classes, activation="softmax")(auxiliary2)

  x = InceptionV1Layer(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Layer(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
  x = InceptionV1Layer(x1=384, reduce_x3=192, x3=384, reduce_x5=48, x5=128, pool_proj=128)(x)
  x = AveragePooling2D(pool_size=7, strides=1)(x)
  x = Dropout(rate = 0.4)(x)
  
  
  output = Flatten()(x)
  output = Dense(units = num_classes , activation="softmax")(output)
  return k.Model(inputs=inputs, outputs = [output, auxiliary1, auxiliary2])

