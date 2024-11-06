import warnings

from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from mtl.model.build import build_backbone, build_neck


@SEGMENTORS.register_module()
class RSCoTrSeg(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)[0]
        if neck is not None:
            self.neck = build_neck(neck)[0]
        self._init_decode_head(decode_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head