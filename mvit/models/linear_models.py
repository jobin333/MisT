import torch

class LinearModel(torch.nn.Module):
    def __init__(self, neurons = [768, 7]):
        super().__init__()
        self.layers = []
        for i in range(len(neurons)-1):
            self.layers.append(torch.nn.Linear(neurons[i], neurons[i+1]))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
