from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from sklearn import preprocessing
from glob import glob

class NumpyDatasetGroupSplit():
    def __init__(self, x, y, group, test_size=.20, n_splits=1, random_state=7):
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
        split = splitter.split(x, groups=group)
        train_inds, test_inds = next(split)
        print("x, y train: ", x[train_inds].shape, y[train_inds].shape)
        print("x, y test: ", x[test_inds].shape, y[test_inds].shape)
        self.split = CSVDataset(x[train_inds], y[train_inds]), CSVDataset(x[test_inds], y[test_inds])
    
    def getSplit(self):
        return self.split
        
class CSVDatasetGroupSplit():
    def __init__(self, x, y, group, test_size=.20, n_splits=1, random_state=7):
        #xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1)
        #self.x_data = torch.from_numpy(xy[:, :-2]) # size [n_samples, n_features]
        #self.y_data = torch.from_numpy(xy[:, [-2]])
        #xy[:, :-2] = preprocessing.MinMaxScaler().fit_transform(xy[:, :-2])'
        _x = x.to_numpy()
        _y = y.to_numpy()
        _group = group.to_numpy()
        #self.groups = torch.from_numpy(group) 
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
        split = splitter.split(_x, groups=_group)
        train_inds, test_inds = next(split)
        print("train, test", _x[train_inds].shape, _x[test_inds].shape)
        self.split = CSVDataset(_x[train_inds], _y[train_inds]), CSVDataset(_x[test_inds], _y[test_inds])

    def getSplit(self):
        return self.split


class CSVDataset(Dataset):
    def __init__(self, x, y):
        # Initialize data, download, etc.
        # read with numpy or pandas
        #xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1)

        # Number of samples
        self.n_samples = x.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(x) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(y) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
# Returns array of files with 'file_type' from 'file_path'
def load_files(file_path, file_type):
    return sorted(glob(f'{file_path}*{file_type}'))