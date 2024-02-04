import torch

def generate_cumsum(x):
    y = x.argmax(-1)
    y = torch.nn.functional.one_hot(y, 7)
    y = torch.cumsum(y, 0)
    return y

class CumsumModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim*2, dim)

    def forward(self, x):
        with torch.no_grad():
            z = generate_cumsum(x) / len(x)
        x = torch.cat((x, z), dim=-1)
        x = self.linear(x)
        return x
    
class CumsumModel2(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim*2, dim)
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        with torch.no_grad():
            z = generate_cumsum(x) / len(x)
        x = torch.cat((x, z), dim=-1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        