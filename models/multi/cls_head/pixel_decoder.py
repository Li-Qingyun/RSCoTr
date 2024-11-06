# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, ModuleList

from mmdet.core.anchor import MlvlPointGenerator


@PLUGIN_LAYERS.register_module()
class MlvlClsPixelDecoder(BaseModule):
    def __init__(self,
                 num_encoder_levels=4,
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 num_outs=4,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_encoder_levels = num_encoder_levels
        self.postional_encoding = build_positional_encoding(
            positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)
        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self):
        """Initialize weights."""
        normal_init(self.level_encoding, mean=0, std=1)

    def forward(self, encoder, neck_feats):
        num_input_levels = len(neck_feats)
        batch_size = neck_feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = num_input_levels - i - 1
            feat_projected = neck_feats[level_idx]
            h, w = feat_projected.shape[-2:]

            # no padding
            padding_mask_resized = feat_projected.new_zeros(
                (batch_size, ) + feat_projected.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat_projected.shape[-2:], level_idx, device=feat_projected.device)
            # normalize
            factor = feat_projected.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            encoder_input_list.append(
                feat_projected.flatten(2).permute(2, 0, 1))
            padding_mask_list.append(
                padding_mask_resized.flatten(1))
            level_positional_encoding_list.append(
                level_pos_embed.flatten(2).permute(2, 0, 1))
            spatial_shapes.append(feat_projected.shape[-2:])
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=0)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_query, batch_size, c)
        memory = encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]
        return outs