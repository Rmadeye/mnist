import numpy as np
import torch
def calculate_accuracy(y, y_pred):
    return (y == y_pred).sum()/len(y)   


class Monitor:
    # batch: dict(loss = 0, acc = 0)

    def __init__(self):
        self.data = dict(loss = list(torch.Tensor([1000,])), 
                         acc = list(torch.tensor([0.0])), best_acc = 0.0,
                            epochs_with_no_improvement = 0, best_epoch = np.nan  
                         )
        self.batch = dict(loss = list(), acc = list())
        self.val = dict(correct = 0, total = 0)

    @staticmethod
    def calculate_accuracy(y, y_pred):
        return (y == y_pred).sum()/len(y)

    def show_last(self, measure: str):
        return self.data[measure][-1]
    
    def evaluate(self):
        
        for key, value in self.batch.items():
            measure = sum(value)/len(value)
            self.data[key].append(measure)
        self.clean()
        
    def track_history(self, measure: str):
        return {epoch: value for epoch, value in enumerate(self.data[measure])}
    
    def validation_results(self):
        return self.val['correct']/self.val['total']*100

    
    def clean(self):
        self.batch = dict(loss = list(), acc = list())