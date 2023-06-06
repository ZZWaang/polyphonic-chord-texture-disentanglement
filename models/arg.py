import copy
from typing import Union, Callable, Optional

import torch.nn as nn
import torch
import math

from torch import Tensor
from torch.nn import functional as F, ModuleList
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm


class PolydisARGDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout):
        super(PolydisARGDecoder).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = PolydisARGDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.transformer_decoder = PolydisARGDecoderModule(decoder_layer, num_layers)

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)

        output = self.transformer_decoder(tgt, tgt_mask, tgt_key_padding_mask)

        output = self.out(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PolydisARGDecoderModule(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(PolydisARGDecoderModule, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class PolydisARGDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PolydisARGDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(PolydisARGDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
