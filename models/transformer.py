import torch
import torch.nn as nn
import math
from einops import rearrange

from models.base_model import ModuleWithDistortion

# switch off flash attention and memory efficient attention to avoid issues with backward pass with vmap and hessian calc
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class PositionalEncoding(nn.Module):
    # This class is from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        if batch_first:
            pe = rearrange(pe, "s b f -> b s f")

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            if batch_first is True:
                x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            else:
                x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.batch_first:
            # Works for both (batch, seq, feat) and (seq, feat) under vmap
            x = x + self.pe[0, :x.shape[-2], :]  # use shape[-2] instead of size(1)
        else:
            x = x + self.pe[:x.shape[0], 0, :]
        return self.dropout(x)


class IdentityNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x):
        return x


class TransfEncoder(ModuleWithDistortion):
    def __init__(self, n_blocks, feat=1, d_model=32, n_out=1, dim_ff=32, nhead=2, p=0.0, identity_norm=False):
        super(TransfEncoder, self).__init__()
        self.in_dim = feat
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_out = n_out

        self.pos_encoder = PositionalEncoding(d_model=d_model, batch_first=True, dropout=p) # No dropout
        self.in_encoding = nn.Linear(feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_ff, nhead=nhead,
                                                batch_first=True, dropout=p)

        if identity_norm:
            # for debug purposes the layer norm can be commented out
            enc_layer.norm1 = IdentityNorm()
            enc_layer.norm2 = IdentityNorm()

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, n_out)

    def forward(self, x):
        no_batch = False
        if len(x.shape) < 3:
            if self.in_dim == 1:
                # (batch, seq_length) -> (batch, features, seq_length)
                x = x.unsqueeze(1)  # Add a feature dimension
            else:
                no_batch = True

        if no_batch:
            # x shape is (features, seq_length) -> (seq_length, features)
            x = rearrange(x, "f s -> s f")
        else:
            # x shape is (batch, features, seq_length) -> (batch, seq_length, features)
            x = rearrange(x, "b f s -> b s f")

        x = self.in_encoding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=-2)  # keep the mean at -2 for the cases when batch dimension is not passed
        x = self.fc(x)  # (batch, n_out)

        return x
