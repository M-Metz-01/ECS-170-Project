

#imports
from local_code.stage_3_code.Method_CNN import Method_CNN


# ORL CNN: bigger images but very few of them (9 per person), so the model stays small and uses dropout
class Method_CNN_ORL(Method_CNN):
    in_channels = 1                       #only the R channel is used (R=G=B for these images)
    image_size = (112, 92)                #image height and width
    num_classes = 40                      #one class per person
    conv_channels = [16, 32, 64]          #3 conv blocks: 112x92 -> 56x46 -> 28x23 -> 14x11
    kernel_size = 3                       #3x3 filters
    pooling = 'max'                       #2x2 max pooling after each block
    hidden_sizes = [128]                  #one fully connected hidden layer
    dropout = 0.3                         #dropout in the fully connected layer (helps with only 360 images)
    max_epoch = 30                        #more epochs since each epoch is only 360 images
    batch_size = 16                       #small batches -> more weight updates per epoch
    learning_rate = 1e-3                  #adam learning rate
