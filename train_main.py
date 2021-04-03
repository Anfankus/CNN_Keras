import tensorflow as tf

from dataset import Dataset
from models.simple_cnn import SimpleCNN
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from models.inception import InceptionV1
from trainer import Trainer

if __name__ == "__main__":
    input_shape = (1,224,224,3)
    num_classes = 1000

    mydata = Dataset()
    mynet = InceptionV1(num_classes=num_classes)

    # if not mynet.built:
    #     mynet.build(input_shape)
    # print(mynet.summary())
    # tf.keras.utils.plot_model(mynet,to_file="plots/InceptionV1.png", show_shapes=True)
    
    mytrainer = Trainer(mynet)
    mytrainer.train(mydata,epoches=5)
