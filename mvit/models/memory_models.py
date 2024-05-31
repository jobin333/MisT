import torch

class MultiLevelMemoryModel(torch.nn.Module):
    '''
    Expecting batch first format (Batch, Sequence, Features)
    Stack of F(z)
    
    '''
    def __init__(self, predictor_model, stack_length, roll_count,
                 number_path, path_multiplier, dropout=0.0): ## roll_count ->  lookback interval
        super().__init__()
        self.stack_length = stack_length
        self.rolls = path_multiplier**torch.arange(number_path)*roll_count
        self.rolls = self.rolls.to(torch.int)
        self.predictor_model = predictor_model
        self.disable_gradient(self.predictor_model)
        self.output_projection2 = torch.nn.Linear(number_path*64, 7)
        self.projections = [torch.nn.Linear(stack_length*7, 64) for i in range(number_path)]
        for i in range(number_path):
            self.reg_params(f'number_path_{i}', self.projections[i])
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
    
    def disable_gradient(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def reg_params(self, name, module):
        for i, param in enumerate(module.parameters()):
            self.register_parameter(name + str(i), param)
    
    def generate_stack(self, x, stack_length, rolls):
        if len(x.shape) == 3:  ## Already stacked
            return x
        stack_list = [x]
        for i in range(stack_length-1):
            shifted_x = torch.roll(stack_list[-1], rolls, dims=0)
            shifted_x[:rolls] = 0.
            stack_list.append(shifted_x)
        stack = torch.stack(stack_list).permute(1,0,2)
        return stack


    def forward(self, x, attn_mask=None):
        with torch.no_grad():
            x = self.predictor_model(x)
            x = self.softmax(x)
            xs = []
            for i in self.rolls:
                xx = self.generate_stack(x, self.stack_length, i.item())
                xx = self.flatten(xx)
                xs.append(xx) 
        ys = []
        for i, x in enumerate(xs):
            y = self.projections[i](x)
            ys.append(y)
        output = torch.concat(ys, -1)
        output = self.relu(output)
        output = self.output_projection2(output)
        return output
