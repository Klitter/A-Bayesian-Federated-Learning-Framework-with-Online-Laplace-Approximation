from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from cnn_lab.autoaugment import Cutout, CIFAR10Policy
import random

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # self.data = [self.dataset[idx] for idx in self.idxs]
        # self.targets = [self.dataset.targets[idx] for idx in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img, label = self.dataset[self.idxs[item]]
        return img, label #self.data[item], self.targets[item]

def get_client_alpha(train_set_group):
    client_n_sample = [len(ts.idxs) for ts in train_set_group]
    total_n_sample = sum(client_n_sample)
    client_alpha = [n_sample / total_n_sample for n_sample in client_n_sample]
    # print(f'alpha = {client_alpha}')
    return client_alpha

def dirichlet_data(data_name, num_users=10, alpha = 100):

    if data_name == 'mnist':
        dataset = datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


    elif data_name == 'cifar10':
        dataset = datasets.CIFAR10('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR10('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif data_name == 'cifar100':

        dataset = datasets.CIFAR100('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    else:
        print ('Data name error')
        return None


    class_num = 10

    dict_users = {i: np.array([]) for i in range(num_users)}

    idxs = np.arange(len(dataset.targets))
    labels = np.asarray(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    class_lableidx = [idxs_labels[:, idxs_labels[1, :] == i][0, :] for i in range(class_num)]

    sample_matrix = np.random.dirichlet([alpha for _ in range(num_users)], class_num).T
    class_sampe_start = [0 for i in range(class_num)]

    def sample_rand(rand, class_sampe_start):
        class_sampe_end = [start + int(len(class_lableidx[sidx]) * rand[sidx]) for sidx, start in enumerate(class_sampe_start)]
        rand_set = np.array([])
        for eidx, rand_end in enumerate(class_sampe_end):
            rand_start = class_sampe_start[eidx]
            if rand_end<= len(class_lableidx[eidx]):
                rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:rand_end]], axis=0)

            else:
                if rand_start< len(class_lableidx[eidx]):
                    rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:]],axis=0)
                else:
                    rand_set=np.concatenate([rand_set,random.sample(class_lableidx[eidx] , rand_end - rand_start +1)],axis=0)
        if rand_set.shape[0] == 0:
            rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
        return rand_set, class_sampe_end

    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_sampe_start)
        dict_users[i] = rand_set

    return [DatasetSplit(deepcopy(dataset), dict_users[i]) for i in range(num_users)], test_dataset




if __name__ == '__main__':
    pass













