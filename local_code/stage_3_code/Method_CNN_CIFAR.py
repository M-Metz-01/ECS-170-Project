

#imports
from local_code.stage_3_code.Method_CNN import Method_CNN


# CIFAR CNN: colored, harder images, so the model is deeper (2 convs per block) and uses batch norm
class Method_CNN_CIFAR(Method_CNN):
    in_channels = 3                       #RGB -> 3 channels
    image_size = (32, 32)                 #image height and width
    num_classes = 10                      #10 object classes
    conv_channels = [32, 64, 128]         #3 conv blocks: 32x32 -> 16x16 -> 8x8 -> 4x4
    convs_per_block = 2                   #2 conv layers before each pooling layer
    kernel_size = 3                       #3x3 filters
    pooling = 'max'                       #2x2 max pooling after each block
    batch_norm = True                     #batch norm after each conv layer
    hidden_sizes = [256]                  #one fully connected hidden layer
    dropout = 0.3                         #dropout in the fully connected layer
    max_epoch = 20                        #num of training epochs
    batch_size = 128                      #num of images per mini-batch
    learning_rate = 1e-3                  #adam learning rate
    lr_schedule = 'cosine'                #slowly lower the learning rate to 0 over training
