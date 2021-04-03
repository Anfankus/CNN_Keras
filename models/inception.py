"""
@cite Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.
"""

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D

class InceptionLayer(k.layers.Layer):
  def __init__(self, x1, reduce_x3, x3, reduce_x5, x5, pool_proj):
    super(InceptionLayer,self).__init__()
    self.x1 = x1
    self.reduce_x3 = reduce_x3
    self.x3 = x3
    self.reduce_x5 = reduce_x5
    self.x5 = x5
    self.pool_proj = pool_proj
  def call(self, x):
    conv_path1 = Conv2D(self.x1, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")(x)

    conv_path2 = Conv2D(self.reduce_x3, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")(x)
    conv_path2 = Conv2D(self.x3, (3,3), (1,1), "same", activation="relu", kernel_initializer="normal")(conv_path2)

    conv_path3 = Conv2D(self.reduce_x5, (1,1), (1,1), "same", activation="relu", kernel_initializer="normal")(x)
    conv_path3 = Conv2D(self.x5, (5,5), (1,1), "same", activation="relu", kernel_initializer="normal")(conv_path3)

    conv_path4 = MaxPooling2D((3,3),(1,1),"same")(x)
    conv_path4 = Conv2D(self.pool_proj,(1,1),(1,1), "same", activation="relu", kernel_initializer="normal")(conv_path4)

    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])


class InceptionV1(k.Model):
  def __init__(self, num_classes):
    super(InceptionV1,self).__init__()
    self.num_classes = num_classes
    
  def call(self, x, training = True):
    x = Conv2D(filters=64, kernel_size=7, strides=(2,2), activation="relu", padding="same", kernel_initializer="normal")(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)
    x = Conv2D(filters=192, kernel_size=3, strides=1, activation="relu",  padding="same", kernel_initializer="normal")(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = InceptionLayer(x1=64, reduce_x3=96, x3=128, reduce_x5=16, x5=32, pool_proj=32)(x)
    x = InceptionLayer(x1=128, reduce_x3=128, x3=192, reduce_x5=32, x5=96, pool_proj=64)(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = InceptionLayer(x1=192, reduce_x3=96, x3=208, reduce_x5=16, x5=48, pool_proj=64)(x)
    x = InceptionLayer(x1=160, reduce_x3=112, x3=224, reduce_x5=24, x5=64, pool_proj=64)(x)
    x = InceptionLayer(x1=128, reduce_x3=128, x3=256, reduce_x5=24, x5=24, pool_proj=64)(x)
    x = InceptionLayer(x1=112, reduce_x3=144, x3=288, reduce_x5=32, x5=64, pool_proj=64)(x)
    x = InceptionLayer(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = InceptionLayer(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
    x = InceptionLayer(x1=384, reduce_x3=192, x3=384, reduce_x5=48, x5=128, pool_proj=128)(x)
    x = AveragePooling2D(pool_size=7, strides=1)(x)
    x = Dropout(rate = 0.4)(x, training=training)
    x = Flatten()(x)
    x = Dense(units = self.num_classes , activation="softmax")(x)

    return x

