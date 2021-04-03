import tensorflow.keras as k
import tensorflow as tf
import numpy as np

class Dataset:
  def __init__(self):
    mnist = k.datasets.mnist
    cifar100 = k.datasets.cifar100
    cifar10 = k.datasets.cifar10

    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    if len(x_train.shape) == 3:
      x_train = np.expand_dims(x_train,axis=-1)
      x_test = np.expand_dims(x_test,axis=-1)

    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train_frame = np.zeros((x_train.shape[0],224,224,x_train.shape[3]))
    x_train_frame[:,:x_train.shape[1],:x_train.shape[2],:] = x_train

    x_test_frame = np.zeros((x_test.shape[0],224,224,x_train.shape[3]))
    x_test_frame[:,:x_test.shape[1],:x_test.shape[2],:] = x_test

    # x_train = x_train_frame.astype('float32')
    # x_test = x_test_frame.astype('float32')

    self.train_ds = tf.data.Dataset.from_tensor_slices((x_train_frame,y_train))
    self.train_ds = self.train_ds.shuffle(buffer_size=10000)
    self.train_ds = self.train_ds.batch(batch_size=5)

    self.test_ds = tf.data.Dataset.from_tensor_slices((x_test_frame,y_test)).batch(batch_size=10)

