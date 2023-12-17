import math

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from mvit.model_utils.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):

    def __init__(self, input_features: int, output_features:int, nhead: int, d_hid: int,
                 nlayers: int, device, dropout: float = 0.5,
                 enable_positional_encoding=True, enable_src_mask=True):
        super().__init__()
        self.model_type = 'Transformer'
        if enable_positional_encoding:
            self.pos_encoder = PositionalEncoding(input_features, dropout)
        encoder_layers = TransformerEncoderLayer(output_features, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(input_features, output_features)
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

        output = self.linear(output)
        return output
