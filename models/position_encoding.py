# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    # NOTE: Use the same positional encoding as the official Transformer repo
    # -------------------------------------------------------------------------

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # NOTE: Reread.
        # ---------------------------------------------------------------------
        
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32) # (batch_size, h, w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # (batch_size, h, w)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # (num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # (num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t # (batch_size, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t # (batch_size, h, w, num_pos_feats)

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # (batch_size, h, w, num_pos_feats)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3) # (batch_size, h, w, num_pos_feats)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # (batch_size, 2 * num_pos_feats, h, w)
        
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()

        # NOTE: (50, num_pos_feats)
        # ---------------------------------------------------------------------
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]

        i = torch.arange(w, device=x.device) # (w)
        j = torch.arange(h, device=x.device) # (h)

        x_emb = self.col_embed(i) # (w, num_pos_feats)
        y_emb = self.row_embed(j) # (h, num_pos_feats)

        # NOTE: Repeat x axis for column embedding and y axis for row embedding
        #       to match the shape of x.
        # ---------------------------------------------------------------------
        pos = torch.cat([
            x_emb
                .unsqueeze(0)     # (1, w, num_pos_feats)
                .repeat(h, 1, 1), # (h, w, num_pos_feats)
            y_emb
                .unsqueeze(1)     # (h, 1, num_pos_feats)
                .repeat(1, w, 1), # (h, w, num_pos_feats)
        ], dim=-1)                # (h, w, 2 * num_pos_feats)

        pos = pos.permute(2, 0, 1) # (2 * num_pos_feats, h, w)

        pos = pos.unsqueeze(0)   # (1, 2 * num_pos_feats, h, w)
        pos = pos.repeat(x.shape[0], 1, 1, 1) # (batch_size, 2 * num_pos_feats, h, w)

        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
