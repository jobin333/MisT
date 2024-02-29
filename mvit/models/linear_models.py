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


class SimpleLinearModel(torch.nn.Module):
    def __init__(self, in_features=768, out_features=7, seq_length=30, enable_sigmoid_out=False):
        super().__init__()
        self.enable_sigmoid_out = enable_sigmoid_out
        self.linear = torch.nn.Linear(in_features=in_features*seq_length, out_features=out_features)
        self.flatten = torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        if self.enable_sigmoid_out:
            x = self.sigmoid(x)
        return x