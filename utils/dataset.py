from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from sklearn import preprocessing

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
    


    '''
    for i in range(10):    
    train_inds, test_inds = next(split)
    unique, counts = np.unique(self.groups[train_inds], return_counts=True)

    _o = dict(zip(unique, counts))
    print("train: ", _o)

    unique, counts = np.unique(self.groups[test_inds], return_counts=True)
    _o = dict(zip(unique, counts))
    print("test: ", _o)
                        
    '''


    '''
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
    split = splitter.split(df, groups=df['group'])
    train_inds, test_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    print(train[:,:])
    '''


    #print("train ", len(train[(train['group'] == _group_id)]))
    #print("test ",len(test[(test['group'] == _group_id)]))
    #print(test['group'])
    

    
    '''
    xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1)
    self.x_data = torch.from_numpy(xy[:, :-2]) # size [n_samples, n_features]
    self.y_data = torch.from_numpy(xy[:, [-2]]) # size [n_samples, 1]
    self.group_data = torch.from_numpy(xy[:, [-1]])
    print(xy.shape)
    gkf = GroupKFold(n_splits=2)

    for train_index, test_index in gkf.split(self.x_data, self.y_data, groups=self.group_data):
        print(self.group_data[train_index].shape, self.group_data[test_index].shape)
        unique, counts = np.unique(self.group_data[train_index], return_counts=True)

        _o = dict(zip(unique, counts))
        print("train: ", _o)

        unique, counts = np.unique(self.group_data[test_index], return_counts=True)
        _o = dict(zip(unique, counts))
        print("test: ", _o)
    '''


    ''' 
    df = pd.read_csv(file_path) 
    #print("df ", df.shape)
    _group_id = 10
    _g = len(df[(df['group'] == _group_id)]['group'])

    print(_g)
    g_k_f =  GroupKFold(n_splits=2)

    for i, (train_index, test_index) in enumerate(g_k_f.split(X, y, groups)):
    '''