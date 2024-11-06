from typing import List

from torch import Tensor

from mmcls.models import GlobalAveragePooling, LinearClsHead, HEADS


@HEADS.register_module()
class SlvlClsHead(LinearClsHead):
    def __init__(self, *args, **kwargs):
        super(SlvlClsHead, self).__init__(*args, **kwargs)
        self.avg_pool = GlobalAveragePooling()

    def pre_logits(self, x: List[Tensor]) -> Tensor:
        cls_token = self.avg_pool(tuple(x))
        if isinstance(cls_token, (tuple, list)):
            cls_token = cls_token[-1]
        return cls_token

    def forward_train(self, neck_feature, backbone_feature, gt_label,
                      shared_encoder, **kwargs):
        return super(SlvlClsHead, self).forward_train(
            backbone_feature, gt_label, **kwargs)

    def simple_test(self, neck_feature, backbone_feature,
                    shared_encoder, softmax=True, post_process=True):
        return super(SlvlClsHead, self).simple_test(
            backbone_feature, softmax, post_process)