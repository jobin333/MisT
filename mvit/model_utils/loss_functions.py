import torch

class SequentialCrossEntropyLoss():
    def __init__(self) -> None:
        self.loss_fn = torch.nn.CrossEntropyLoss() 

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        input = input.squeeze(0)
        loss = self.loss_fn(input, target)
        return loss