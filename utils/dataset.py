import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ComFoldData(Dataset):
    def __init__(self, data, label, com_label, train=True):
        # reading data as table form
        self.images = data
        self.labels = label

        self.comp_labels = com_label

        self.train = train

    def __getitem__(self, index):
        img, target, comp_label = torch.from_numpy(self.images[index]).float(), \
                                  torch.from_numpy(self.labels[index]).float(), \
                                  torch.from_numpy(self.comp_labels[index]).float()

        return img, target, comp_label, index

    def __len__(self):
        return len(self.labels)

'''
fold: fold-th fold of 10-cross fold
nfold: 10
'''
def ComFold(batchsize, Filename, nfold, fold):
    data = np.genfromtxt(Filename[0], delimiter=',')
    label = np.genfromtxt(Filename[1], delimiter=',')
    com_label = np.genfromtxt(Filename[2], delimiter=',')

    n_test = len(com_label)//nfold
    print('n size:', n_test)
    y = np.arange(len(com_label))
    start = fold*n_test
    if start+n_test > len(com_label):
        test = y[start:]
    else:
        test = y[start:start+n_test]
    train = np.setdiff1d(y, test)

    # training dataset
    train_scene = ComFoldData(data[train, :], label[train, :], com_label[train, :], train=True)
    print("train data shape:", data[train, :].shape)
    train_loader = DataLoader(
        train_scene,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # Data loader for test dataset
    size = len(test)
    test_scene = ComFoldData(data[test, :], label[test, :], com_label[test, :], train=False)
    print("test data & label shape:", data[test, :].shape, label[test, :].shape)
    test_loader = DataLoader(
        test_scene,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, test_loader