
#imports
from local_code.base_class.dataset import dataset
import numpy as np
import pickle


#loads one pickle file (ORL, MNIST or CIFAR) and returns the train and test images (X) and labels (y)
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # file to load: 'ORL', 'MNIST' or 'CIFAR'
    normalize = True                 # scale pixel values from [0,255] to [0,1]

    def __init__(self, dName=None, dDescription=None):
        #sets the dataset name and descriptor using the base class
        super().__init__(dName, dDescription)

    def _to_arrays(self, instances):
        #turns a list of {'image', 'label'} dicts into one image array and one label array
        X = np.array([inst['image'] for inst in instances], dtype=np.float32)
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        if self.dataset_source_file_name == 'ORL':
            #ORL is gray-scale stored as RGB with R=G=B, so keep only the R channel
            X = X[:, :, :, 0:1]
            #ORL labels go from 1 to 40; shift them to 0-39 so they work with cross entropy
            y = y - 1
        elif X.ndim == 3:
            #MNIST images have no channel axis (N, 28, 28), so add one -> (N, 28, 28, 1)
            X = X[:, :, :, None]

        #PyTorch conv layers want (N, channels, height, width) instead of (N, height, width, channels)
        X = np.ascontiguousarray(X.transpose(0, 3, 1, 2))

        #scale pixel values from 0-255 to 0-1
        if self.normalize:
            X = X / 255.0
        return X, y

    def load(self):
        #reads the pickle file and returns {'train': {'X', 'y'}, 'test': {'X', 'y'}}
        print('loading data from ' + self.dataset_source_file_name + ' ...')
        #full path to the file
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)

        #convert each split to arrays
        X_train, y_train = self._to_arrays(raw['train'])
        X_test, y_test = self._to_arrays(raw['test'])
        print('train images:', X_train.shape, 'test images:', X_test.shape)

        #return both splits
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}
