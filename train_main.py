import tensorflow as tf

from dataset import Dataset
from models.simple_cnn import SimpleCNN
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from models.inception import InceptionV1, InceptionV3
from trainer import Trainer

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

if __name__ == "__main__":
    image_shape = (299,299,3)
    num_classes = 1000

    mydata = Dataset()
    mynet = InceptionV1(image_shape, num_classes)

    print(mynet.summary())
    tf.keras.utils.plot_model(mynet,to_file="plots/InceptionV1.png",show_shapes=True)
    
    # mytrainer = Trainer(mynet)
    # mytrainer.train(mydata,epoches=5)
