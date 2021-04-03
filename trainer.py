import tensorflow as tf
import tensorflow.keras as k
class Trainer:
  def __init__(self,model:k.Model):
    self.model = model

    self.loss = k.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.optimizer = k.optimizers.Adam()

    self.train_loss_metric = k.metrics.Mean(name="train_loss")
    self.train_acc_metric = k.metrics.SparseCategoricalAccuracy(name="train_acc")

    self.test_loss_metric = k.metrics.Mean(name="test_loss")
    self.test_acc_metric = k.metrics.SparseCategoricalAccuracy(name="test_acc")

  def train(self,dataset, epoches = 10):
    for epoch in range(epoches):
      self.train_loss_metric.reset_states()
      self.train_acc_metric.reset_states()
      
      self.test_loss_metric.reset_states()
      self.test_acc_metric.reset_states()

      for images, labels in dataset.train_ds:
        self.train_step(images,labels)

      for images, labels in dataset.test_ds:
        self.test_step(images,labels)

      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {self.train_loss_metric.result()}, '
        f'Accuracy: {self.train_acc_metric.result() * 100}, '
        f'Test Loss: {self.test_loss_metric.result()}, '
        f'Test Accuracy: {self.test_acc_metric.result() * 100}'
      )

  # @tf.function
  def train_step(self, img, label):
      with tf.GradientTape() as tape:
          predict = self.model(img)
          loss = self.loss(label, predict)
      grads = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

      self.train_loss_metric(loss)
      self.train_acc_metric(label, predict)
  
  @tf.function
  def test_step(self, img, label):
      predict = self.model(img)  
      loss = self.loss(label, predict)

      self.test_loss_metric(loss)
      self.test_acc_metric(label, predict)
