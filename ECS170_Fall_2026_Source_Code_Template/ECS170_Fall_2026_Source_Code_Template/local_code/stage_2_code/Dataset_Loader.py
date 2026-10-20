'''
Concrete IO class for the stage 2 dataset (pre-partitioned train/test CSV files).
Each row: label first, then the feature values (comma separated).
'''

from local_code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # file to load, e.g. 'train.csv'
    label_first = True               # label is the first column
    normalize = True                 # scale pixel values from [0,255] to [0,1]

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data from ' + self.dataset_source_file_name + ' ...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        # skip a header row automatically if the first line is not numeric
        with open(path, 'r') as f:
            first = f.readline().strip().split(',')
        try:
            [float(v) for v in first]
            skip = 0
        except ValueError:
            skip = 1
        arr = np.loadtxt(path, delimiter=',', skiprows=skip, dtype=np.float32)
        if self.label_first:
            y, X = arr[:, 0], arr[:, 1:]
        else:
            y, X = arr[:, -1], arr[:, :-1]
        if self.normalize:
            X = X / 255.0
        return {'X': X.astype(np.float32), 'y': y.astype(np.int64)}
