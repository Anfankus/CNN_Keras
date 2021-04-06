"""
@cite Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.
"""

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization, ReLU, GlobalAvgPool2D

class InceptionV1Module(k.Model):
  def __init__(self, x1, reduce_x3, x3, reduce_x5, x5, pool_proj):
    super(InceptionV1Module,self).__init__()
    self.conv1 = Conv2D(x1, 1, 1, "same", activation="relu", kernel_initializer="normal")

    self.conv2_1 = Conv2D(reduce_x3, 1, 1, "same", activation="relu", kernel_initializer="normal")
    self.conv2_2 = Conv2D(x3, 3, 1, "same", activation="relu", kernel_initializer="normal")

    self.conv3_1 = Conv2D(reduce_x5, 1, 1, "same", activation="relu", kernel_initializer="normal")
    self.conv3_2 = Conv2D(x5, 5, 1, "same", activation="relu", kernel_initializer="normal")

    self.pool4_1 = MaxPooling2D(3,1,"same")
    self.pool4_2 = Conv2D(pool_proj,1,1, "same", activation="relu", kernel_initializer="normal")
    
  def call(self, x):
    conv_path1 = self.conv1(x)
    conv_path2 = self.conv2_2(self.conv2_1(x))
    conv_path3 = self.conv3_2(self.conv3_1(x))
    conv_path4 = self.pool4_2(self.pool4_1(x))
    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])



"""
  InceptionV1 implemention according to 'Going Deeper with Convolutions'

  input_shape : the image shape, default is (224,224,3)
"""
def InceptionV1(input_shape, num_classes):
  inputs = k.Input(shape=input_shape)
  x = Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", padding="same", kernel_initializer="normal")(inputs)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = Conv2D(filters=192, kernel_size=3, strides=1, activation="relu",  padding="same", kernel_initializer="normal")(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Module(x1=64, reduce_x3=96, x3=128, reduce_x5=16, x5=32, pool_proj=32)(x)
  x = InceptionV1Module(x1=128, reduce_x3=128, x3=192, reduce_x5=32, x5=96, pool_proj=64)(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Module(x1=192, reduce_x3=96, x3=208, reduce_x5=16, x5=48, pool_proj=64)(x)

  auxiliary1 = AveragePooling2D(pool_size=5, strides=3)(x)
  auxiliary1 = Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu", kernel_initializer="normal")(auxiliary1)
  auxiliary1 = Flatten()(auxiliary1)
  auxiliary1 = Dense(units=1024, activation="relu")(auxiliary1)
  auxiliary1 = Dropout(rate = 0.7)(auxiliary1)
  auxiliary1 = Dense(units = num_classes, activation="softmax")(auxiliary1)

  x = InceptionV1Module(x1=160, reduce_x3=112, x3=224, reduce_x5=24, x5=64, pool_proj=64)(x)
  x = InceptionV1Module(x1=128, reduce_x3=128, x3=256, reduce_x5=24, x5=24, pool_proj=64)(x)
  x = InceptionV1Module(x1=112, reduce_x3=144, x3=288, reduce_x5=32, x5=64, pool_proj=64)(x)

  auxiliary2 = AveragePooling2D(pool_size=5, strides=3)(x)
  auxiliary2 = Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu", kernel_initializer="normal")(auxiliary2)
  auxiliary2 = Flatten()(auxiliary2)
  auxiliary2 = Dense(units=1024, activation="relu")(auxiliary2)
  auxiliary2 = Dropout(rate = 0.7)(auxiliary2)
  auxiliary2 = Dense(units = num_classes, activation="softmax")(auxiliary2)

  x = InceptionV1Module(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
  x = InceptionV1Module(x1=256, reduce_x3=160, x3=320, reduce_x5=32, x5=128, pool_proj=128)(x)
  x = InceptionV1Module(x1=384, reduce_x3=192, x3=384, reduce_x5=48, x5=128, pool_proj=128)(x)
  x = AveragePooling2D(pool_size=7, strides=1)(x)
  x = Dropout(rate = 0.4)(x)
  
  
  output = Flatten()(x)
  output = Dense(units = num_classes , activation="softmax")(output)
  return k.Model(inputs=inputs, outputs = [output, auxiliary1, auxiliary2])


class Conv2D_BN(k.layers.Layer):
  def __init__(self, filters,kernel_size, strides=1, padding="same"):
    super(Conv2D_BN, self).__init__()
    self.conv = Conv2D(filters, kernel_size, strides, padding, kernel_initializer="normal")
    self.bn = BatchNormalization()
  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return k.activations.relu(x)

class InceptionV3ModuleA(k.layers.Layer):
  def __init__(self, pool_proj):
    super(InceptionV3ModuleA, self).__init__()
    self.conv1 = Conv2D_BN(filters=64, kernel_size=1)

    self.conv2_1 = Conv2D_BN(filters=48, kernel_size=1)
    self.conv2_2 = Conv2D_BN(filters=64, kernel_size=5)

    self.conv3_1 = Conv2D_BN(filters=64, kernel_size=1)
    self.conv3_2 = Conv2D_BN(filters=96, kernel_size=3)
    self.conv3_3 = Conv2D_BN(filters=96, kernel_size=3)

    self.pool4_1 = AveragePooling2D(pool_size=3, strides=1, padding="same")
    self.pool4_2 = Conv2D_BN(filters=pool_proj, kernel_size=1)

  def call(self, x):
    conv_path1 = self.conv1(x)
    conv_path2 = self.conv2_2(self.conv2_1(x))
    conv_path3 = self.conv3_3(self.conv3_2(self.conv3_1(x)))
    conv_path4 = self.pool4_2(self.pool4_1(x))
    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])

class InceptionV3ModuleB(k.layers.Layer):
  def __init__(self, x7):
    super(InceptionV3ModuleB, self).__init__()
    self.conv1_1 = Conv2D_BN(filters=192, kernel_size=1)

    self.conv2_1 = Conv2D_BN(filters=x7, kernel_size=1)
    self.conv2_2 = Conv2D_BN(filters=x7, kernel_size=(7,1))
    self.conv2_3 = Conv2D_BN(filters=192, kernel_size=(1,7))

    self.conv3_1 = Conv2D_BN(filters=x7, kernel_size=1)
    self.conv3_2 = Conv2D_BN(filters=x7, kernel_size=(7,1))
    self.conv3_3 = Conv2D_BN(filters=x7, kernel_size=(1,7))
    self.conv3_4 = Conv2D_BN(filters=x7, kernel_size=(7,1))
    self.conv3_5 = Conv2D_BN(filters=192, kernel_size=(1,7))

    self.pool4_1 = AveragePooling2D(pool_size=3, strides=1, padding="same")
    self.pool4_2 = Conv2D_BN(filters=192, kernel_size=1)

  def call(self, x):
    conv_path1 = self.conv1_1(x)
    conv_path2 = self.conv2_3(self.conv2_2(self.conv2_1(x)))
    conv_path3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(x)))))
    conv_path4 = self.pool4_2(self.pool4_1(x))
    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])

class InceptionV3ModuleC(k.layers.Layer):
  def __init__(self):
    super(InceptionV3ModuleC, self).__init__()
    self.conv1_1 = Conv2D_BN(filters=320, kernel_size=1)
    self.conv2_1 = Conv2D_BN(filters=384, kernel_size=1)
    self.conv2_1a = Conv2D_BN(filters=384, kernel_size=(1,3))
    self.conv2_1b = Conv2D_BN(filters=384, kernel_size=(3,1))

    self.conv3_1 = Conv2D_BN(filters=384, kernel_size=1)
    self.conv3_2 = Conv2D_BN(filters=384, kernel_size=3)
    self.conv3_2a = Conv2D_BN(filters=384, kernel_size=(3,1))
    self.conv3_2b = Conv2D_BN(filters=384, kernel_size=(1,3))

    self.pool4_1 = AveragePooling2D(pool_size=3, strides=1, padding="same")
    self.pool4_2 = Conv2D_BN(filters=192, kernel_size=1)
  
  def call(self, x):
    conv_path1 = self.conv1_1(x)
    
    conv_path2 = self.conv2_1(x)
    conv_path2a = self.conv2_1a(conv_path2)
    conv_path2b = self.conv2_1b(conv_path2)
    conv_path2 = k.layers.concatenate([conv_path2a, conv_path2b])

    conv_path3 = self.conv3_2(self.conv3_1(x))
    conv_path3a = self.conv3_2a(conv_path3)
    conv_path3b = self.conv3_2b(conv_path3)
    conv_path3 = k.layers.concatenate([conv_path3a, conv_path3b])

    conv_path4 = self.pool4_2(self.pool4_1(x))

    return k.layers.concatenate([conv_path1, conv_path2, conv_path3, conv_path4])


"""
  InceptionV3 implemention according to 'Rethinking the Inception Architecture for Computer Vision'

  input_shape : the image shape, default is (299,299,3)
"""

def InceptionV3(input_shape, num_classes):
  inputs = k.Input(shape=input_shape)

  x = Conv2D_BN(filters=32, kernel_size=3, strides=2, padding="valid")(inputs)
  x = Conv2D_BN(filters=32, kernel_size=3, strides=1, padding="valid")(x)
  x = Conv2D_BN(filters=64, kernel_size=3, strides=1, padding="same")(x)
  x = MaxPooling2D(pool_size=3, strides=2)(x)
  x = Conv2D_BN(filters=80, kernel_size=3, strides=1, padding="valid")(x)
  x = Conv2D_BN(filters=192, kernel_size=3, strides=2, padding="valid")(x)
  x = MaxPooling2D(pool_size=3, strides=2)(x)

  # Inception module A
  x = InceptionV3ModuleA(32)(x)
  x = InceptionV3ModuleA(64)(x)
  x = InceptionV3ModuleA(64)(x)

  # Reduction A
  re_a1 = Conv2D_BN(filters=384, kernel_size=3, strides=2, padding="valid")(x)

  re_a2 = Conv2D_BN(filters=64, kernel_size=1)(x)
  re_a2 = Conv2D_BN(filters=96, kernel_size=3)(re_a2)
  re_a2 = Conv2D_BN(filters=96, kernel_size=3, strides=2, padding="valid")(re_a2)

  re_a3 = MaxPooling2D(pool_size=3, strides=2, padding="valid")(x)
  re_a3 = Conv2D_BN(filters=288,kernel_size=1)(re_a3)

  x = k.layers.concatenate([re_a1, re_a2, re_a3])

  # Inception module B
  x = InceptionV3ModuleB(128)(x)
  x = InceptionV3ModuleB(160)(x)
  x = InceptionV3ModuleB(160)(x)
  x = InceptionV3ModuleB(192)(x)

  # auxiliary
  auxiliary = AveragePooling2D(pool_size=5, strides=3)(x)
  auxiliary = Conv2D_BN(filters=128, kernel_size=1)(auxiliary)
  auxiliary = Flatten()(auxiliary)
  auxiliary = Dense(units= 768)(auxiliary)
  auxiliary = BatchNormalization()(auxiliary)
  auxiliary = ReLU()(auxiliary)
  auxiliary = Dense(units= num_classes, activation="softmax")(auxiliary)

  # Reduction B
  ra_b1 = Conv2D_BN(filters=192, kernel_size=1)(x)
  ra_b1 = Conv2D_BN(filters=320, kernel_size=3,strides=2, padding="valid")(ra_b1)

  ra_b2 = Conv2D_BN(filters=192, kernel_size=1)(x)
  ra_b2 = Conv2D_BN(filters=192, kernel_size=(1,7))(ra_b2)
  ra_b2 = Conv2D_BN(filters=192, kernel_size=(7,1))(ra_b2)
  ra_b2 = Conv2D_BN(filters=192, kernel_size=3, strides=2, padding="valid")(ra_b2)

  ra_b3 = MaxPooling2D(pool_size=3,strides=2)(x)
  x = k.layers.concatenate([ra_b1, ra_b2, ra_b3])

  # Inception module C
  x = InceptionV3ModuleC()(x)
  x = InceptionV3ModuleC()(x)

  x = GlobalAvgPool2D()(x)
  x = Flatten()(x)
  x = Dense(units=1000, activation="relu")(x)

  return k.Model(inputs=inputs, outputs = [x, auxiliary])


