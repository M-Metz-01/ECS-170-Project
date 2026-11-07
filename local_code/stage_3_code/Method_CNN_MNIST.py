

#imports
from local_code.stage_3_code.Method_CNN import Method_CNN


# MNIST CNN: two small conv blocks are enough for digits
class Method_CNN_MNIST(Method_CNN):
    in_channels = 1                       #gray-scale -> 1 channel
    image_size = (28, 28)                 #image height and width
    num_classes = 10                      #digits 0-9
    conv_channels = [32, 64]              #2 conv blocks: 28x28 -> 14x14 -> 7x7
    kernel_size = 3                       #3x3 filters
    pooling = 'max'                       #2x2 max pooling after each block
    hidden_sizes = [128]                  #one fully connected hidden layer
    dropout = 0.25                        #dropout in the fully connected layer
    max_epoch = 10                        #num of training epochs
    batch_size = 128                      #num of images per mini-batch
    learning_rate = 1e-3                  #adam learning rate
