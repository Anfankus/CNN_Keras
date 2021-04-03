import tensorflow as tf
import tensorflow.keras as k

class SimpleCNN(k.Model):
  def __init__(self, num_classes):
    super(SimpleCNN,self).__init__()
    self.conv1 = k.layers.Conv2D(32,3,activation="relu")
    self.flatten = k.layers.Flatten()
    self.fc1 = k.layers.Dense(32,activation="relu")
    self.fc2 = k.layers.Dense(num_classes)
  # def build(input_shape):
  #   return super().build(input_shape)
  def call(self,x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    return x