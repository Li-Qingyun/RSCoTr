import warnings
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors import DETR, SingleStageDetector

from mtl.model.build import build_backbone, build_neck


@DETECTORS.register_module()
class DINO(DETR):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)[0]
        if neck is not None:
            self.neck = build_neck(neck)[0]
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
