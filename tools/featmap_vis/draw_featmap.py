# Modified from mmengine
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mmengine.visualization.utils import img_from_canvas


def draw_featmap(featmap: torch.Tensor,
                 overlaid_image: Optional[np.ndarray] = None,
                 channel_reduction: Optional[str] = 'squeeze_mean',
                 topk: int = 20,
                 arrangement: Tuple[int, int] = (4, 5),
                 resize_shape: Optional[tuple] = None,
                 channel_id: Optional[int] = None,
                 alpha: float = 0.5,
                 with_text: bool = True) -> np.ndarray:
    """Draw featmap.

    - If `overlaid_image` is not None, the final output image will be the
    weighted sum of img and featmap.

    - If `resize_shape` is specified, `featmap` and `overlaid_image`
    are interpolated.

    - If `resize_shape` is None and `overlaid_image` is not None,
    the feature map will be interpolated to the spatial size of the image
    in the case where the spatial dimensions of `overlaid_image` and
    `featmap` are different.

    - If `channel_reduction` is "squeeze_mean" and "select_max",
    it will compress featmap to single channel image and weighted
    sum to `overlaid_image`.

    -  if `channel_reduction` is None

      - If topk <= 0, featmap is assert to be one or three
      channel and treated as image and will be weighted sum
      to ``overlaid_image``.
      - If topk > 0, it will select topk channel to show by the sum of
      each channel. At the same time, you can specify the `arrangement`
      to set the window layout.

    Args:
        featmap (torch.Tensor): The featmap to draw which format is
            (C, H, W).
        overlaid_image (np.ndarray, optional): The overlaid image.
            Default to None.
        channel_reduction (str, optional): Reduce multiple channels to a
            single channel. The optional value is 'squeeze_mean'
            or 'select_max'. Defaults to 'squeeze_mean'.
        topk (int): If channel_reduction is not None and topk > 0,
            it will select topk channel to show by the sum of each channel.
            if topk <= 0, tensor_chw is assert to be one or three.
            Defaults to 20.
        arrangement (Tuple[int, int]): The arrangement of featmap when
            channel_reduction is not None and topk > 0. Defaults to (4, 5).
        resize_shape (tuple, optional): The shape to scale the feature map.
            Default to None.
        channel_id (int, optional): To return the specific channel_id.
        alpha (Union[int, List[int]]): The transparency of featmap.
            Defaults to 0.5.
        with_text (bool, optional): Defaults to False.

    Returns:
        np.ndarray: RGB image.
    """
    assert isinstance(featmap,
                      torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                      f' but got {type(featmap)}')
    assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                              f'but got {featmap.ndim}'
    featmap = featmap.detach().cpu()

    if overlaid_image is not None:
        if overlaid_image.ndim == 2:
            overlaid_image = cv2.cvtColor(overlaid_image,
                                          cv2.COLOR_GRAY2RGB)

        if overlaid_image.shape[:2] != featmap.shape[1:]:
            warnings.warn(
                f'Since the spatial dimensions of '
                f'overlaid_image: {overlaid_image.shape[:2]} and '
                f'featmap: {featmap.shape[1:]} are not same, '
                f'the feature map will be interpolated. '
                f'This may cause mismatch problems ！')
            if resize_shape is None:
                featmap = F.interpolate(
                    featmap[None],
                    overlaid_image.shape[:2],
                    mode='bilinear',
                    align_corners=False)[0]

    if resize_shape is not None:
        featmap = F.interpolate(
            featmap[None],
            resize_shape,
            mode='bilinear',
            align_corners=False)[0]
        if overlaid_image is not None:
            overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

    if channel_reduction is not None:
        assert channel_reduction in [
            'squeeze_mean', 'select_max'], \
            f'Mode only support "squeeze_mean", "select_max", ' \
            f'but got {channel_reduction}'
        if channel_reduction == 'select_max':
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, 1)
            feat_map = featmap[indices]
        else:
            feat_map = torch.mean(featmap, dim=0)
        return convert_overlay_heatmap(feat_map, overlaid_image, alpha,
                                       gray_look=overlaid_image is None)
    elif topk <= 0:
        featmap_channel = featmap.shape[0]
        assert featmap_channel in [
            1, 3
        ], ('The input tensor channel dimension must be 1 or 3 '
            'when topk is less than 1, but the channel '
            f'dimension you input is {featmap_channel}, you can use the'
            ' channel_reduction parameter or set topk greater than '
            '0 to solve the error')
        return convert_overlay_heatmap(featmap, overlaid_image, alpha,
                                       gray_look=overlaid_image is None)
    elif channel_id is not None:
        featmap_channel = featmap.shape[0]
        assert channel_id < featmap_channel
        return convert_overlay_heatmap(featmap[channel_id], overlaid_image,
                                       alpha, gray_look=overlaid_image is None)
    else:
        row, col = arrangement
        channel, height, width = featmap.shape
        assert row * col >= topk, 'The product of row and col in ' \
                                  'the `arrangement` is less than ' \
                                  'topk, please set the ' \
                                  '`arrangement` correctly'

        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = featmap[indices]

        fig = plt.figure(frameon=False)
        # Set the window layout
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        dpi = fig.get_dpi()
        fig.set_size_inches((width * col + 1e-2) / dpi,
                            (height * row + 1e-2) / dpi)
        for i in range(topk):
            axes = fig.add_subplot(row, col, i + 1)
            axes.axis('off')
            if with_text:
                axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
            axes.imshow(
                convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                        alpha, gray_look=overlaid_image is None))
        image = img_from_canvas(fig.canvas)
        plt.close(fig)
        return image


def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5,
                            gray_look: bool = False) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    if gray_look:
        # heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_BONE)
        heat_img = norm_img[..., None].repeat(3, -1)
    else:
        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img