import numpy as np
import sklearn
import torch

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def generate_compl_labels(train_labels, num_com):
    '''
    :param train_labels: true labels
    :param num_com: the number of complementary labels per instance
    :return: generated complementary labels for each instance
    '''
    k = np.size(train_labels, 1)
    n = np.size(train_labels, 0)
    comp_Y = np.zeros([n, k])
    labels_hat = np.array(1 - train_labels, dtype=bool)  # ensure true labels can't be selected
    for idx in range(n):
        candidates = np.arange(k).reshape(1, k)
        for i in range(num_com):
            mask = labels_hat[idx].reshape(1, -1)
            candidates_ = candidates[mask]
            index = np.random.randint(0, len(candidates_))
            comp_Y[idx][candidates_[index]] = 1
            labels_hat[idx, candidates_[index]] = False  # expecting the selected complementary label
    return comp_Y


# generate complementary labels
files = ["data/scene_label.csv", "data/yeast_label.csv"]
com_file = ["data/scene_com_label.csv", "data/yeast_com_label.csv"]

for i in range(len(files)):
    label = np.genfromtxt(files[i], delimiter=',')
    print(label.shape)
    com_labels = generate_compl_labels(label, 1)
    np.savetxt(com_file[i], com_labels, delimiter=',')
