import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, BatchNormalization,Activation

class Conv2D_BN(k.layers.Layer):
  def __init__(self, filters,kernel_size, strides=1, padding="same", activation="relu"):
    super(Conv2D_BN, self).__init__()
    self.conv = Conv2D(filters, kernel_size, strides, padding, kernel_initializer="normal")
    self.bn = BatchNormalization()
    self.act = Activation(activation=activation)
  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x
