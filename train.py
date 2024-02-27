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
from scripts.metrices import Monitor

r =  lambda x: round(x, 2)

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

    X, X_val, y, y_val = train_test_split(
        data.drop('label', axis=1), data['label'], test_size=0.1, random_state=training_params['seed'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=training_params['seed'])
    train_loader = prepare_data(torch.tensor(X_train.values), torch.tensor(y_train.values), training_params['batch_size'])
    test_loader = prepare_data(torch.tensor(X_test.values), torch.tensor(y_test.values), training_params['batch_size'])
    val_loader = prepare_data(torch.tensor(X_val.values), torch.tensor(y_val.values), training_params['batch_size'])

    net = ConvNet(**net_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=training_params['lr'])
    metrics = Monitor()
    acc_calculator = Monitor.calculate_accuracy
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
            metrics.batch['loss'].append(loss.item())
        net.eval()
        # metrics.data['loss'].append(loss.item())
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = net(inputs.float().unsqueeze(1))
                predicted = torch.argmax(outputs, dim=1)  # Apply sigmoid and round
                metrics.batch['acc'].append(acc_calculator(labels, predicted))
        metrics.evaluate()
        print(f'Epoch {epoch}, '
      f'Accuracy: {metrics.show_last("acc")}, '
      f'Loss: {metrics.show_last("loss")}, '
      f'Best accuracy: {metrics.data["best_acc"]}, '
      f'Epochs with no improvement: {metrics.data["epochs_with_no_improvement"]}',
      f'Best epoch: {metrics.data["best_epoch"]}')

        if metrics.show_last("acc") <  metrics.data['best_acc']:
            metrics.data['epochs_with_no_improvement'] += 1 
            if metrics.data['epochs_with_no_improvement'] == 5:
                print('Early stopping')
                break
        else:
            metrics.data['best_acc'] = metrics.show_last("acc")
            metrics.data['epochs_with_no_improvement'] = 0
            torch.save(net.state_dict(), 'weights/model.pt')
            metrics.data['best_epoch'] = epoch
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = net(inputs.float().unsqueeze(1))
            predicted = torch.argmax(outputs, dim=1)  # Apply sigmoid and round
            metrics.val['total'] += labels.size(0)
            metrics.val['correct'] += (predicted == labels).sum().item()
    print(f'Validation accuracy: {r(metrics.validation_results())}')  
        

train_model('data/input/train.csv', list(range(0,10)), 'hparams/base.yaml')



    

    
    


