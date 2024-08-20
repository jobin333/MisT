import math
import torch

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from mvit.model_utils.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):

    '''
        Note:
            Mandatory property of dataset: batch_first = True
    '''

    def __init__(self, input_features: int, output_features:int, nhead: int, d_hid: int,
                 nlayers: int, device, seq_length:int,  dropout: float = 0.5,
                 enable_positional_encoding=True, enable_src_mask=True):
        super().__init__()
        self.output_features = output_features
        self.model_type = 'Transformer'
        if enable_positional_encoding:
            self.pos_encoder = PositionalEncoding(input_features, dropout)
        encoder_layers = TransformerEncoderLayer(input_features, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_features*seq_length, output_features)
        self.device = device
        self.enable_positional_encoding = enable_positional_encoding
        self.enable_src_mask = enable_src_mask

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if self.enable_positional_encoding:
            src = self.pos_encoder(src)
        if src_mask is None and self.enable_src_mask:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
            output = self.transformer_encoder(src, src_mask)

        else:
            output = self.transformer_encoder(src)

        output = self.flatten(output)
        output = self.linear(output)
        return output


class MultiLevelTransformerMemoryModel(torch.nn.Module):
    '''
    Expecting batch first format (Batch, Sequence, Features)
    Stack of F(z)
    
    '''
    def __init__(self, stack_length, strides, dropout, nhead, dim_feedforward, 
                 num_layers, num_surg_phase, dmodel, device=None): 
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        
        self.dmodel=dmodel
        self.stack_length = stack_length
        self.strides = torch.tensor(strides)
        self.strides = self.strides.to(torch.int)
        self.number_path = len(self.strides)
        self.seq_length = len(self.strides)
        self.num_surg_phase = num_surg_phase

        self.mhat_reducer = torch.nn.Linear(self.num_surg_phase*stack_length, self.dmodel)
        self.output_projection = torch.nn.Linear(self.seq_length*self.dmodel, num_surg_phase)
        
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(-1)
        self.relu = torch.nn.ReLU()
        self.positional_encoding = self.generate_positional_encoding()
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dmodel, 
                                                                          nhead=nhead, batch_first=True,
                                                             dim_feedforward=dim_feedforward )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, 
                                                               num_layers=num_layers)
        

            

    def generate_positional_encoding(self,):
        position = torch.arange(self.seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dmodel, 2) * (-math.log(10000.0) / self.dmodel))
        pe = torch.zeros(self.seq_length, 1, self.dmodel) # max_len = seq_length
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.to(self.device)
        return pe.transpose(1,0)
        
    
    def disable_gradient(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def reg_params(self, name, module):
        for i, param in enumerate(module.parameters()):
            self.register_parameter(name + str(i), param)
    
    def generate_stack(self, x, stack_length, strides):
        if len(x.shape) == 3:  ## Already stacked
            return x
        stack_list = [x]
        for i in range(stack_length-1):
            shifted_x = torch.roll(stack_list[-1], strides, dims=0)
            if strides > 0:
                shifted_x[:strides] = 0.
            else:
                shifted_x[strides:] = 0.
            stack_list.append(shifted_x)
        stack = torch.stack(stack_list).permute(1,0,2)
        return stack


    def forward(self, x, attn_mask=None):
        with torch.no_grad():
            x = self.softmax(x)
            xs = []
            for i in self.strides:
                xx = self.generate_stack(x, self.stack_length, i.item())
                xs.append(xx) 
        ys = []
        for stride, x in zip(self.strides, xs):
            y = self.flatten(x)
            y = self.mhat_reducer(y)
            ys.append(y)
        output = torch.stack(ys).transpose(1,0)
        output = self.relu(output)
        output = self.transformer_encoder(output)
        output = self.flatten(output)
        output = self.output_projection(output)
        return output
