'''
Concrete IO class for the stage 2 dataset (pre-partitioned train/test CSV files).
Each row: label first, then the feature values (comma separated).
'''

#imports
from local_code.base_class.dataset import dataset
import numpy as np

#loads one CSV file (train or test) and returns its images (X) and labels (y)/
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # file to load, e.g. 'train.csv'
    label_first = True               # label is the first column
    normalize = True                 # scale pixel values from [0,255] to [0,1]

    def __init__(self, dName=None, dDescription=None):
        #sets the dataset name and descriptor using the base class
        super().__init__(dName, dDescription)

    def load(self):
        #reads the CSV file and returns {'X': pixel values, 'y': labels}
        print('loading data from ' + self.dataset_source_file_name + ' ...')
        #full path to the file
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        # skip a header row automatically if the first line is not numeric
        #read the first line of the file and split it into values
        with open(path, 'r') as f:
            first = f.readline().strip().split(',')
        #try converting every value in the first ine to a number
        try:
            [float(v) for v in first]
            skip = 0         #all numbers -> no header, skip nothing
        except ValueError:
            skip = 1         #found text -> the first line is a header, skip it
        #read the whole file into one numpy array
        arr = np.loadtxt(path, delimiter=',', skiprows=skip, dtype=np.float32)
        #split each row into the label and the pixel values
        if self.label_first:

            y, X = arr[:, 0], arr[:, 1:]
        else:
            y, X = arr[:, -1], arr[:, :-1]
        #scale pixel values from 0-255 to 0-1
        if self.normalize:
            X = X / 255.0
        #return the pixels as floats and the labels as ints
        return {'X': X.astype(np.float32), 'y': y.astype(np.int64)}
