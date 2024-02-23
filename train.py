import os
import argparse
from typing import Optional, List

import torch
from torch import nn
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split   
import yaml 

from network.model import ConvNet
from network.loader import prepare_data

def train_model(data_path: str, limit_digits: Optional[List[int]], hparams: str ):
    #
    data = pd.read_csv(data_path)
    if limit_digits:
        data = data[data['label'].isin(limit_digits)].reset_index(drop=True)
        data['label'] = data['label'].apply(lambda x: limit_digits.index(x))

    with open(hparams, 'rt') as fp:
        hparams = yaml.safe_load(fp)
    
    net_params = hparams['network']
    training_params = hparams['training']

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('label', axis=1), data['label'], test_size=0.2, random_state=training_params['seed'])
    train_loader = prepare_data(torch.tensor(X_train.values), torch.tensor(y_train.values), training_params['batch_size'])
    test_loader = prepare_data(torch.tensor(X_test.values), torch.tensor(y_test.values), training_params['batch_size'])

    net = ConvNet(**net_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=training_params['lr'])

    for epoch in range(training_params['num_epochs']):
        net.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # print(labels)
            optimizer.zero_grad()
            outputs = net(inputs.reshape(-1, 1, 28, 28).float())
            # breakpoint()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = net(inputs.float().unsqueeze(1))
                predicted = torch.argmax(outputs, dim=1)  # Apply sigmoid and round
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch}, Accuracy: {100 * correct / total}, loss: {loss.item()}')

        

train_model('data/input/train.csv', [1,2,3,4,5], 'hparams/base.yaml')



    

    
    


