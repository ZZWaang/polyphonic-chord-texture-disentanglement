# %%
import math
import copy
import warnings
import re
import sys
import glob
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from torch.utils.data import Dataset, DataLoader

# %%
num_layers = 6
learning_rate = 0.0002
epoch = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len, device=device):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model, device=device)
        positions_list = torch.arange(0, max_len, dtype=torch.float, device=device).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2, device=device).float() * (
            -math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


# %%
class zTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
            self,
            dim_model,
            num_heads,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=4 * dim_model,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.layer,
            num_layers=num_decoder_layers,
        )

    def get_tgt_mask(self, size, device=device) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask.to(device)

    def forward(
            self,
            input_tensor,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        input_tensor = input_tensor * math.sqrt(self.dim_model)
        input_tensor = self.positional_encoder(input_tensor)
        tgt_mask = self.get_tgt_mask(input_tensor.shape[-2])
        zero_tensor = torch.zeros_like(input_tensor, requires_grad=True, device=device)

        # Transformer blocks - Out size = (batch_size, sequence length, dim_model)
        transformer_out = self.transformer(zero_tensor, input_tensor, tgt_mask=tgt_mask)

        return transformer_out


# %%
class InfoNCELoss(nn.Module):
    def __init__(self, input_dim, sample_dim, skip_projection=False):
        super().__init__()
        self.input_dim = input_dim
        self.sample_dim = sample_dim
        self.skip_projection = skip_projection
        if skip_projection:
            if input_dim != sample_dim:
                raise ValueError('Hidden sizes do not match.')
        else:
            self.projection = nn.Linear(input_dim, sample_dim, bias=False)

    def forward(self, input, positive, negative, temperature):
        # input: (bs, input_dim)
        # positive: (bs, n_p, sample_dim)
        # negative: (bs, n_n, sample_dim)
        # temperature: float number

        if not self.skip_projection:
            input = self.projection(input)
        p_logits = torch.einsum('bij,bj->bi', positive, input)
        n_logits = torch.einsum('bij,bj->bi', negative, input)
        p_logits = torch.sum(torch.exp(p_logits / temperature), dim=-1)
        n_logits = torch.sum(torch.exp(n_logits / temperature), dim=-1)
        loss = torch.div(p_logits, p_logits + n_logits)
        loss = - torch.log(loss)
        return torch.mean(loss)


# %%
model = zTransformer(
    dim_model=512, num_heads=8, num_decoder_layers=num_layers, dropout_p=0.1
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)
loss_fn.cuda()
