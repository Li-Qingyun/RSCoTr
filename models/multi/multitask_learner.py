import warnings
from collections import OrderedDict

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

import mmcv
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         MultiScaleDeformableAttention)
from mmcv.cnn import MODELS

from mmcls.models.utils.augment import Augments
from mmdet.core import bbox2result
from mmseg.core import add_prefix
from mmseg.ops import resize

from mmdet.core.visualization import color_val_matplotlib

from mtl.model.build import build_backbone, build_neck, build_head


supported_tasks = ('cls', 'det', 'seg')


@MODELS.register_module()
class MTL(BaseModule):
    PALETTE = None
    def __init__(self,
                 backbone,
                 neck,
                 shared_encoder,
                 cls_head=None,
                 bbox_head=None,
                 seg_head=None,
                 task_weight=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MTL, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.shared_encoder = build_transformer_layer_sequence(shared_encoder)

        self.task_weight = dict(cls=1, det=1, seg=1)
        if task_weight is not None:
            assert isinstance(task_weight, dict)
            self.task_weight.update(task_weight)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        cls_augments_cfg = train_cfg['cls'].get('augments', None)
        if cls_augments_cfg is not None:
            self.cls_augments = Augments(cls_augments_cfg)
        bbox_head.update(train_cfg=train_cfg['det'])
        bbox_head.update(test_cfg=test_cfg['det'])
        # seg_head.update(train_cfg=train_cfg['seg'])
        # seg_head.update(test_cfg=test_cfg['seg'])
        self.task_pretrain = self.train_cfg.get('task_pretrain', None)

        self.cls_head = build_head(cls_head, 'mmcls')
        self.bbox_head = build_head(bbox_head, 'mmdet')
        self.seg_head = build_head(seg_head, 'mmseg')

    def init_weights(self) -> None:
        super(MTL, self).init_weights()
        # init_weights defined in MultiScaleDeformableAttention
        for layer in self.shared_encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention):
                    attn.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        backbone_feature = self.backbone(img)
        neck_feature = self.neck(backbone_feature[-3:])
        return neck_feature, backbone_feature

    def forward_train(self, task: str, *args, **kwargs):
        assert task in supported_tasks
        return getattr(self, f'forward_train_{task}')(*args, **kwargs)

    def forward_test(self,
                     task: 'str, List[str]',
                     img: 'Tensor, List[Tensor]',
                     img_metas: list,
                     *args,
                     **kwargs):
        if isinstance(task, list):
            task = list(set(task))
            if len(task) == 1:
                task = task[0]
            else:
                raise NotImplementedError(
                    'The current implementation only '
                    'support same task in a batch')
        if isinstance(img, list):
            num_augs = len(img)
            if num_augs != 1:
                raise NotImplementedError(
                    'The current implementation does not support TTA ')
            img = img[0]
        if isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        return self.simple_test(task, img, img_metas, *args, **kwargs)

    def simple_test(self, task: 'str', *args, **kwargs):
        assert task in supported_tasks
        return getattr(self, f'simple_test_{task}')(*args, **kwargs)

    def forward_train_cls(self, img, gt_label, **kwargs):
        if self.cls_augments is not None:
            img, gt_label = self.cls_augments(img, gt_label)
        neck_feature, backbone_feature = self.extract_feat(img)
        losses = dict()
        loss = self.cls_head.forward_train(
            neck_feature, backbone_feature, gt_label, self.shared_encoder)
        losses.update(loss)
        return losses

    def forward_train_det(self, img, img_metas, gt_bboxes, gt_labels,
                          gt_bboxes_ignore=None):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)[0]
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,
                                              self.shared_encoder)
        return losses

    def forward_train_seg(self, img, img_metas, gt_semantic_seg):
        neck_feature, backbone_feature = self.extract_feat(img)
        losses = dict()
        loss_decode = self.seg_head.forward_train(
            neck_feature, backbone_feature, img_metas,
            gt_semantic_seg, self.shared_encoder)
        losses.update(add_prefix(loss_decode, 'seg'))
        return losses

    def simple_test_cls(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        neck_feature, backbone_feature = self.extract_feat(img)
        res = self.cls_head.simple_test(
            neck_feature, backbone_feature,
            shared_encoder=self.shared_encoder, **kwargs)
        return res

    def simple_test_det(self, img, img_metas, rescale=False):
        batch_size = len(img_metas)
        for img_id in range(batch_size):
            img_metas[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        feat = self.extract_feat(img)[0]
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale,
            shared_encoder=self.shared_encoder)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def whole_inference_seg(self, img, img_meta, rescale):
        neck_feature, backbone_feature = self.extract_feat(img)
        seg_logit = self.seg_head.forward_test(
            neck_feature, backbone_feature, img_meta, self.shared_encoder)
        seg_logit = resize(
            input=seg_logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.seg_head.align_corners)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.seg_head.align_corners,
                warning=False)
        return seg_logit

    def inference_seg(self, img, img_meta, rescale):
        assert self.test_cfg['seg'].mode in ['whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg['seg'].mode == 'whole':
            seg_logit = self.whole_inference_seg(img, img_meta, rescale)
        else:
            raise NotImplementedError
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test_seg(self, img, img_meta, rescale=True):
        seg_logit = self.inference_seg(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        task = data.get('task', None)
        dataset_name = data.get('dataset_name', None)
        log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')

        if hasattr(self, 'task_weight'):
            weight = self.task_weight[task]
            loss *= weight
            log_vars = {k: v * weight for k, v in log_vars.items()}

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        task = data.get('task', None)
        dataset_name = data.get('dataset_name', None)
        log_vars = add_prefix(log_vars, f'{task}.{dataset_name}')

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img',))
    def forward(self, task, img, img_metas, return_loss=True,
                dataset_name=None, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(
                task=task, img=img, img_metas=img_metas, **kwargs)
        else:
            return self.forward_test(
                task=task, img=img, img_metas=img_metas, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def load_task_pretrain(self):

        def get_mapped_name(name: str) -> str:
            new_name = name
            if new_name.startswith('bbox_head.transformer.encoder'):
                new_name = new_name.replace('bbox_head.transformer.encoder',
                                            'shared_encoder')
            return new_name

        def mapping_state_dict(state_dict: OrderedDict) -> OrderedDict:
            out = OrderedDict()
            for name, param in state_dict.items():
                new_name = get_mapped_name(name)
                assert new_name not in out, f'{name}-->{new_name}'
                out[new_name] = param
            return out

        def delete_neck_convs_bias(state_dict: OrderedDict) -> OrderedDict:
            out = OrderedDict()
            for name, param in state_dict.items():
                if name.startswith('neck') and name.endswith('conv.bias'):
                    continue
                out[name] = param
            return out

        if self.task_pretrain is None:
            print('#######################################\n'
                  'You did not set task_pretrain, hence it is skipped.'
                  '#######################################\n')
            return

        rule = self.task_pretrain.get('rule', None)
        pretrainded = self.task_pretrain['pretrained']
        # dm: -> dino_mmdet
        sd = torch.load(pretrainded)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if rule == 'dino_mmdet':
            sd = delete_neck_convs_bias(sd)
            sd = mapping_state_dict(sd)
        incompatiblekeys = self.load_state_dict(sd, strict=False)
        print('#######################################\n'
              f'load task pretrain of rule:{rule}\n'
              f'ckpt path: {pretrainded}\n'
              f'incompatiblekeys: {incompatiblekeys}\n'
              '#######################################\n')

    def show_result(self, img, pred, *args, **kwargs):
        if 'pred_class' in pred:  # cls
            return self.show_cls_result(img, pred, *args, **kwargs)
        elif isinstance(pred, list):
            if len(pred) > 1 or isinstance(pred[0], list):
                pred = pred[0] if isinstance(pred[0], list) else pred
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                kwargs['show_with_gt'] = 'annotation' in kwargs
                return self.show_det_result(img, pred, *args, **kwargs)
            elif pred[0].ndim > 1:
                return self.show_seg_result(img, pred, *args, **kwargs)
            else:
                return self.show_cls_result(img, pred, *args, **kwargs)
        else:
            raise ValueError()

    def show_cls_result(self,
                        img,
                        result,
                        **kwargs):
        if isinstance(result, list):
            result = result[0]
        if isinstance(result, dict):
            assert 'pred_class' in result
            result = result['pred_class']
        # print(np.argmax(result))

    def show_det_result(self,
                        img,
                        result,
                        score_thr=0.3,
                        bbox_color=(255, 110, 110),
                        text_color='black',
                        # text_color=(72, 101, 241),
                        mask_color=None,
                        thickness=2,
                        font_size=15,
                        win_name='',
                        show=False,
                        wait_time=0,
                        out_file=None,
                        show_with_gt=True,
                        annotation=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes  # Modified by LQY
        kwargs = dict(class_names=self.CLASSES['dior'],
                      score_thr=score_thr,
                      thickness=thickness,
                      font_size=font_size,
                      win_name=win_name,
                      show=show,
                      wait_time=wait_time,
                      out_file=out_file)
        if not show_with_gt:
            args = [img, bboxes, labels, segms]
            kwargs["bbox_color"] = bbox_color
            kwargs["text_color"] = text_color
            kwargs["mask_color"] = mask_color
            imshow_results = imshow_det_bboxes
        else:
            args = [img, annotation, result]
            det_color = tuple([v for v in bbox_color[::-1]])
            kwargs["det_bbox_color"] = det_color
            kwargs["det_text_color"] = 'black'
            kwargs["det_mask_color"] = det_color
            gt_color = tuple([v * 255 for v in (0.09,0.78,1)[::-1]])
            kwargs["gt_bbox_color"] = gt_color
            kwargs["gt_text_color"] = gt_color
            kwargs["gt_mask_color"] = gt_color
            kwargs["face_alpha"] = 0.9
            imshow_results = imshow_gt_det_bboxes
        img = imshow_results(*args, **kwargs)

        if not (show or out_file):
            return img

    def show_seg_result(self,
                        img,
                        result,
                        palette=None,
                        win_name='',
                        show=False,
                        wait_time=0,
                        out_file=None,
                        **kwargs):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        opacity = 1
        """
        opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        """

        CLASSES = self.CLASSES['potsdam']

        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if 5 in seg:
            seg += (seg == 5).astype(seg.dtype) * -5
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None,
                      with_text=True,
                      face_color='black',
                      face_alpha=0.4):  # Modified by LQY
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    myfont = fm.FontProperties(fname='./times.ttf')

    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)
    # text_color = (0,0,0)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    EPS = 1e-2
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        if with_text:
            ax.text(
                bbox_int[0],
                # bbox_int[1],
                bbox_int[1],
                f'{label_text}',
                bbox={
                    'facecolor': face_color,
                    'alpha': face_alpha,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color=text_color,
                fontsize=font_size,
                verticalalignment='bottom',
                horizontalalignment='left',
                fontproperties = myfont)
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(255, 102, 61),
                         gt_text_color=(255, 102, 61),
                         gt_mask_color=(255, 102, 61),
                         det_bbox_color=(72, 101, 241),
                         det_text_color=(72, 101, 241),
                         det_mask_color=(72, 101, 241),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None,
                         face_color='black',
                         face_alpha=0.4):
    """General visualization GT and result function.

    Args:
      img (str or ndarray): The image to be displayed.)
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'
      result (tuple[list] or list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown.  Default: 0
      gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (255, 102, 61)
      det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (72, 101, 241)
      thickness (int): Thickness of lines. Default: 2
      font_size (int): Font size of texts. Default: 13
      win_name (str): The window name. Default: ''
      show (bool): Whether to show the image. Default: True
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
         Default: None

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(
        result,
        (tuple, list)), f'Expected tuple or list, but get {type(result)}'

    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False,
        with_text=False)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = mask_util.decode(segms)
        segms = segms.transpose(2, 0, 1)

    # img = imshow_gt_det_bboxes(
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file,
        face_color=tuple([v / 255.0 for v in det_bbox_color[::-1]] + [1.]),
        face_alpha=face_alpha)
    return img


if __name__ == '__main__':
    from mmcv import Config
    config = Config.fromfile('configs/multi/DINO-MTL_swin-t-p4-w7_1x1.py')
    model = MTL(**config.model)