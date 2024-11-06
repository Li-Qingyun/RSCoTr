import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmcv.cnn import build_plugin_layer, constant_init

from mmcls.models.builder import HEADS
from mmcls.models.heads.linear_head import LinearClsHead
from mmcls.models.necks.gap import GlobalAveragePooling


@HEADS.register_module()
class MlvlClsHead(LinearClsHead):
    def __init__(self,
                 *args,
                 pixel_decoder=None,
                 scheme: int = 5,  # do not set 0(test mode)
                 **kwargs):
        super(MlvlClsHead, self).__init__(*args, **kwargs)
        self.pixel_decoder = pixel_decoder
        self.scheme = scheme
        self._init_layers()

    def _init_layers(self):
        if not isinstance(self.pixel_decoder, BaseModule):
            self.pixel_decoder = build_plugin_layer(self.pixel_decoder)[1]

        ### init modules of specific scheme
        if self.scheme in [1, 2, 4, 8]:
            self.avg_pool = GlobalAveragePooling()
        if self.scheme in [5, 6, 7]:
            feat_length = {5: (4,), 6: (7,), 7: (4, 7, 14, 28)}
            in_channels = sum(x**2 for x in feat_length[self.scheme])
            self.out_proj = nn.Linear(in_channels, 1)
            constant_init(self.out_proj, 1 / in_channels)
        if self.scheme in [8]:
            num_encoder_levels = self.pixel_decoder.num_encoder_levels
            self.out_proj = nn.Linear(num_encoder_levels, 1)
            constant_init(self.out_proj, 1 / num_encoder_levels)

    def pre_logits(self, mlvl_feats):
        """
        Input:
            mlvl_feats:
            List[Tensor[16, 256, 4, 4],
                 Tensor[16, 256, 7, 7],
                 Tensor[16, 256, 14, 14],
                 Tensor[16, 256, 28, 28]]
        Output:
            cls_token:
            Tensor[16, 256]
        """
        return getattr(self, f'pre_logits_{self.scheme}')(mlvl_feats)

    def forward_encoder(self, encoder, neck_feature, backbone_feature):
        return self.pixel_decoder(encoder, neck_feature)

    def forward_train(self, neck_feature, backbone_feature, gt_label,
                      shared_encoder, **kwargs):
        x = self.forward_encoder(
            shared_encoder, neck_feature, backbone_feature)
        return super(MlvlClsHead, self).forward_train(x, gt_label, **kwargs)

    def simple_test(self, neck_feature, backbone_feature,
                    shared_encoder, softmax=True, post_process=True):
        x = self.forward_encoder(
            shared_encoder, neck_feature, backbone_feature)
        return super(MlvlClsHead, self).simple_test(x, softmax, post_process)

    # test the schemes
    def pre_logits_0(self, mlvl_feats):
        self.test_pre_logits_implementations(mlvl_feats)

    ### SCHEMES
    ## 1. without weight
    # 1.1 adopt one level and avg_pool
    def pre_logits_1(self, mlvl_feats):
        INDEX = 0
        cls_token = self.avg_pool(mlvl_feats[INDEX])
        return cls_token

    def pre_logits_2(self, mlvl_feats):
        INDEX = 1
        cls_token = self.avg_pool(mlvl_feats[INDEX])
        return cls_token

    # 1.2 adopt all levels and avg_pool
    def pre_logits_3(self, mlvl_feats):
        mlvl_seq = [feat.flatten(2) for feat in mlvl_feats]
        mlvl_seq = torch.cat(mlvl_seq, dim=2)
        cls_token = torch.adaptive_avg_pool1d(mlvl_seq, 1).squeeze()
        return cls_token

    # 1.3 avg_pool each level, output their mean
    def pre_logits_4(self, mlvl_feats):
        mlvl_token = self.avg_pool(tuple(mlvl_feats))
        cls_token = sum(mlvl_token) / len(mlvl_token)
        return cls_token

    ## 2. with weight
    # 2.1 adopt one level and linear proj tokens
    def pre_logits_5(self, mlvl_feats):
        INDEX = 0
        feat = mlvl_feats[INDEX].flatten(2)
        cls_token = self.out_proj(feat).squeeze()
        return cls_token

    def pre_logits_6(self, mlvl_feats):
        INDEX = 1
        feat = mlvl_feats[INDEX].flatten(2)
        cls_token = self.out_proj(feat).squeeze()
        return cls_token

    # 2.2 adopt all levels and linear all tokens
    def pre_logits_7(self, mlvl_feats):
        mlvl_seq = [feat.flatten(2) for feat in mlvl_feats]
        mlvl_seq = torch.cat(mlvl_seq, dim=2)
        cls_token = self.out_proj(mlvl_seq).squeeze()
        return cls_token

    # 2.3 avg_pool each level, linear proj mlvl_token
    def pre_logits_8(self, mlvl_feats):
        mlvl_token = self.avg_pool(tuple(mlvl_feats))
        mlvl_token = torch.stack(mlvl_token, -1)
        cls_token = self.out_proj(mlvl_token).squeeze()
        return cls_token

    def test_pre_logits_implementations(self, mlvl_feats=None):
        if mlvl_feats is None:
            mlvl_feats = [
                torch.rand((16, 256, x, x)).cuda() for x in (4, 7, 14, 28)]
        bs = mlvl_feats[0].size(0)
        outs = []
        for i in range(1, 9):
            self.scheme = i
            self._init_layers()
            self.cuda()
            out = getattr(self, f'pre_logits_{i}')(mlvl_feats)
            error_msg = 'pre_logits_{}: {}'
            assert isinstance(out, torch.Tensor), error_msg.format(i, type(out))
            assert out.size(0) == bs, error_msg.format(i, out.size(0))
            assert out.size(1) == 256, error_msg.format(i, out.size(1))
            assert len(out.shape) == 2, error_msg.format(i, len(out.shape))
            outs.append(out)
            print('Correct !')
        return outs