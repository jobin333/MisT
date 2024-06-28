import torch
from torch.nn import  TransformerEncoderLayer, TransformerEncoder

class MultiLevelMemoryModel(torch.nn.Module):
    '''
    Expecting batch first format (Batch, Sequence, Features)
    Stack of F(z)
    
    '''
    def __init__(self, predictor_model, stack_length, roll_count,
                 number_path, path_multiplier, dropout=0.0, num_surg_phase=7, roll_start_with_one=True, rolls=None): ## roll_count ->  lookback interval
        super().__init__()
        self.stack_length = stack_length
        if roll_start_with_one:
            self.rolls = path_multiplier**torch.arange(number_path - 1)*roll_count
            self.rolls = torch.cat([self.rolls, torch.ones(1)])
        else:
            self.rolls = path_multiplier**torch.arange(number_path)*roll_count

        if rolls is not None:
            self.rolls = rolls
        self.rolls = self.rolls.to(torch.int)
        self.predictor_model = predictor_model
        self.disable_gradient(self.predictor_model)
        self.output_projection2 = torch.nn.Linear(number_path*64, num_surg_phase)
        self.projections = [torch.nn.Linear(stack_length*num_surg_phase, 64) for i in range(number_path)]
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



class MultiLevelTransformerMemoryModel(torch.nn.Module):
    '''
    Expecting batch first format (Batch, Sequence, Features)
    Stack of F(z)
    
    '''
    def __init__(self, predictor_model, stack_length, rolls,
                 dropout=0.0, nhead=8, dim_feedforward=1024, num_layers=6,  num_surg_phase=7,
                 dmodel=128): 
        super().__init__()
        self.stack_length = stack_length
        self.rolls = torch.tensor(rolls)
        self.rolls = self.rolls.to(torch.int)
        self.number_path = len(self.rolls)
        self.seq_length = len(self.rolls)
        self.predictor_model = predictor_model
        self.disable_gradient(self.predictor_model)

        self.output_projection2 = torch.nn.Linear(self.seq_length*dmodel, num_surg_phase)
        self.projections = [torch.nn.Linear(
            num_surg_phase*stack_length, dmodel) for i in range(self.number_path)]
        
        for i in range(self.number_path):
            self.reg_params(f'number_path_{i}', self.projections[i])
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(-1)
        self.relu = torch.nn.ReLU()
        
        transformer_encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead, 
                                                            dropout=dropout, batch_first=True,
                                                             dim_feedforward=dim_feedforward )
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer,
                                                       num_layers=num_layers)
    
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
            if rolls > 0:
                shifted_x[:rolls] = 0.
            else:
                shifted_x[rolls:] = 0.
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
        output = torch.stack(ys).transpose(1,0)
        output = self.flatten(output)
        output = self.relu(output)
        output = self.output_projection2(output)
        return output
