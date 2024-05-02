import argparse

import torch
from torch import nn, optim
from torchsummary import summary

from model import CNNModel
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--save_path', required=True)
    p.add_argument('--indices', type=str, default = 'None')
    p.add_argument('--gpu_id', type = int, default = 0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type = float, default = 0.8)
    p.add_argument('--batch_size', type = int, default=64)
    p.add_argument('--n_epochs', type = int, default = 20)
    
    config = p.parse_args()
    
    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d'%config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)

    print('Train:', len(train_loader.dataset))
    print('Valid:', len(valid_loader.dataset))
    print('Test:', len(test_loader.dataset))

    model = CNNModel(10).to(device)
    print(summary(model, (3,32,32), config.batch_size))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(config)
    trainer.train(model, criterion, optimizer, train_loader, valid_loader, test_loader, device)

if __name__ == '__main__':
    config = define_argparser()
    main(config)