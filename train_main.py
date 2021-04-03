import tensorflow as tf

from dataset import Dataset
from models.simple_cnn import SimpleCNN
from models.alexnet import AlexNet
from models.vgg import VGG16, VGG19
from models.inception import InceptionV1
from trainer import Trainer

if __name__ == "__main__":
    image_shape = (224,224,3)
    input_shape = (1,224,224,3)
    num_classes = 1000

    mydata = Dataset()
    mynet = SimpleCNN(num_classes)

    if mynet.built:
        print(mynet.summary())
    else :
        # 测试模型对于数据尺寸的正确性
        mynet.build(input_shape) 
    
    mytrainer = Trainer(mynet)
    mytrainer.train(mydata,epoches=5)
