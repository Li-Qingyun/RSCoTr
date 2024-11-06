import mmcls.models, mmdet.models, mmseg.models
from mmcv.runner import BaseModule
import mmcv.cnn.bricks.transformer as transformer
from mmdet.models.utils import build_transformer as mmdet_build_transformer


def build_backbone(backbone, base_mm: str = 'mmdet',
                   return_init_requirement: bool = False):
    init_requirement = False
    if not isinstance(backbone, BaseModule):
        init_requirement = True
        if base_mm == 'mmcls':
            backbone = mmcls.models.builder.build_backbone(backbone)
        elif base_mm == 'mmdet':
            backbone = mmdet.models.builder.build_backbone(backbone)
        elif base_mm == 'mmseg':
            backbone = mmseg.models.builder.build_backbone(backbone)
        else:
            raise NotImplementedError(f'Not support base_mm {base_mm}')
    if return_init_requirement:
        return backbone, init_requirement
    else:
        return backbone


def build_neck(neck, base_mm: str = 'mmdet',
               return_init_requirement: bool = False):
    init_requirement = False
    if not isinstance(neck, BaseModule):
        init_requirement = True
        if base_mm == 'mmcls':
            neck = mmcls.models.builder.build_backbone(neck)
        elif base_mm == 'mmdet':
            neck = mmdet.models.builder.build_backbone(neck)
        elif base_mm == 'mmseg':
            neck = mmseg.models.builder.build_backbone(neck)
        else:
            raise NotImplementedError(f'Not support base_mm {base_mm}')
    if return_init_requirement:
        return neck, init_requirement
    else:
        return neck


def build_head(head, base_mm: str = 'mmdet',
               return_init_requirement: bool = False):
    init_requirement = False
    if not isinstance(head, BaseModule):
        init_requirement = True
        if base_mm == 'mmcls':
            head = mmcls.models.builder.build_head(head)
        elif base_mm == 'mmdet':
            head = mmdet.models.builder.build_head(head)
        elif base_mm == 'mmseg':
            head = mmseg.models.builder.build_head(head)
        else:
            raise NotImplementedError(f'Not support base_mm {base_mm}')
    if return_init_requirement:
        return head, init_requirement
    else:
        return head


def build_transformer(transformer, return_init_requirement: bool = False):
    init_requirement = False
    if not isinstance(transformer, BaseModule):
        init_requirement = True
        transformer = mmdet_build_transformer(transformer)
    if return_init_requirement:
        return transformer, init_requirement
    else:
        return transformer


def build_transformer_layer_sequence(transformer_layer_sequence,
                                     return_init_requirement: bool = False):
    if not isinstance(transformer_layer_sequence, BaseModule):
        module = transformer.build_transformer_layer_sequence(
            transformer_layer_sequence)
        init_requirement = True
    else:
        module = transformer_layer_sequence
        init_requirement = False
    if return_init_requirement:
        return module, init_requirement
    else:
        return module

