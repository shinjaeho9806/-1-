import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pickle

class Cifar10Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

def load_cifar10(is_train = True):

    dataset = datasets.CIFAR10(root = './data', train = is_train, download = True)

    x = dataset.data / 255.
    y = dataset.targets
    return x, y

def get_loaders(config):
    x, y = load_cifar10(is_train = True)
    
    if config.indices == 'None':
        train_cnt = x.shape[0] * config.train_ratio
        train_cnt = int(train_cnt)
        valid_cnt = x.shape[0] - train_cnt

        indices = torch.randperm(x.shape[0])
        indices = indices.numpy()
        np.save('indices.npy', indices)
        indices = torch.from_numpy(indices)
    else:
        train_cnt = x.shape[0] * config.train_ratio
        train_cnt = int(train_cnt)
        valid_cnt = x.shape[0] - train_cnt
        indices = np.load(config.indices)
        indices = torch.from_numpy(indices)
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(np.array(y))
    
    train_x, valid_x = torch.index_select(
        x,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)

    mean_r, mean_g, mean_b = torch.mean(train_x, axis = (0,1,2)).numpy()
    std_r, std_g, std_b = torch.std(train_x, axis = (0,1,2)).numpy()
    print('mean_r : {:.4f}, mean_g : {:.4f}, mean_b : {:.4f}'.format(mean_r, mean_g, mean_b))
    print('std_r : {:.4f}, std_g : {:.4f}, std_b : {:.4f}'.format(std_r, std_g, std_b))

    mean = (mean_r, mean_g, mean_b)
    std = (std_r, std_g, std_b)

    standard_dict = {}
    standard_dict['mean'] = mean
    standard_dict['std'] = std

    with open('standard_dict.pkl', 'wb') as f:
        pickle.dump(standard_dict,f)

    def _train_collate_fn(dataset):
        x_batch, y_batch = [], []
        for x, y in dataset:
            x = transforms.ToTensor()(x)
            x = transforms.Normalize(mean, std)(x)
            x = transforms.RandomHorizontalFlip(0.5)(x)
            #x = transforms.RandomRotation(30)(x)
            y = torch.Tensor([y])
            x_batch.append(x)
            y_batch.append(y)
        x_batch = torch.stack(x_batch).float()
        y_batch = torch.cat(y_batch).long()
        return (x_batch,y_batch)
    
    def _valid_collate_fn(dataset):
        x_batch, y_batch = [], []
        for x, y in dataset:
            x = transforms.ToTensor()(x)
            x = transforms.Normalize(mean, std)(x)
            y = torch.Tensor([y])
            x_batch.append(x)
            y_batch.append(y)
        x_batch = torch.stack(x_batch).float()
        y_batch = torch.cat(y_batch).long()
        return (x_batch,y_batch)

    train_y, valid_y = torch.index_select(
        y,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)

    train_loader = DataLoader(
        dataset = Cifar10Dataset(train_x.numpy(), train_y.numpy()),
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = _train_collate_fn,
    )

    valid_loader = DataLoader(
        dataset = Cifar10Dataset(valid_x.numpy(), valid_y.numpy()),
        batch_size = config.batch_size,
        shuffle = False,
        collate_fn = _valid_collate_fn,
    )

    test_x, test_y = load_cifar10(is_train = False)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(np.array(test_y))
    test_loader = DataLoader(
        dataset = Cifar10Dataset(test_x.numpy(), test_y.numpy()),
        batch_size = config.batch_size,
        shuffle = False,
        collate_fn = _valid_collate_fn,
    )


    return train_loader, valid_loader, test_loader



