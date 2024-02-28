import numpy as np
import torch
import yaml
import  pandas as pd

from network.model import ConvNet
from network.loader import prepare_data

def predict(data: np.ndarray, model_path: str, hparams: str = 'hparams/base.yaml', device: str = 'cpu') -> np.ndarray:

    with open(hparams, 'rt') as fp:
        hparams = yaml.safe_load(fp)
    net_params = hparams['network']
    # metrics = hparams['metrics']
    print(f"Loading model from {model_path} with parameters {hparams['network']}")
    net = ConvNet(**net_params)
    model_data = torch.load(model_path)
    net.load_state_dict(model_data['state_dict'])
    net.eval()
    loader = prepare_data(torch.tensor(data), torch.zeros(data.shape[0]), 1)
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = net(inputs.reshape(-1, 1, 28, 28).float())
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
            
    df = pd.DataFrame({'ImageId' : list(range(1, len(predictions)+1)),
                       'Label': np.array(predictions)})
    df.to_csv('predictions.csv', index=False)
    return  df


if __name__ == '__main__':
    data = pd.read_csv('data/input/test.csv').values
    model_path = 'weights/model.pt'
    predictions = predict(data, model_path)
    print(predictions)