# Copyright (c) OpenMMLab. All rights reserved.
import sys
import torch
import torch.nn as nn
# from mmcv.cnn import ConvModule
import warnings
import torch.nn.functional as F
sys.path.append('/Users/lly/codes/DL_Project/')
from model.backbone.convnext import get_convnext

# from mmseg.registry import MODELS
# from ..utils import resize
# from .decode_head import BaseDecodeHead
# from .psp_head import PPM

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            # 计算池化窗口大小和步幅
            # input_height, input_width = 16, 16
            # output_height, output_width = pool_scale, pool_scale
            # kernel_size = (input_height // output_height, input_width // output_width)
            # stride = kernel_size
            self.append(
                nn.Sequential(
                    # 使用 AvgPool2d
                    # nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBlocks(self.in_channels, self.channels, kernel_size=1, norm_type='batch', act_type='relu')
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

def _transform_inputs(inputs, in_index=[-1], input_transform='multiple_select', align_corners=False):
        """Transform inputs for decoder. 选择输入特征图列表中的某些层

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if input_transform == 'resize_concat':
            inputs = [inputs[i] for i in in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif input_transform == 'multiple_select':
            inputs = [inputs[i] for i in in_index]
        else:
            inputs = inputs[in_index]

        return inputs


class ConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', act_type='relu', kernel_size=3, stride=1, padding=0, dialation=1):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dialation)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UPerHead(torch.nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels=[128, 256, 512, 1024], channels=512, dropout_ratio=0.1, num_classes=6, align_corners=False, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.pool_scales = pool_scales

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners)
        self.bottleneck = ConvBlocks(in_channels=self.in_channels[-1] + len(pool_scales) * self.channels, out_channels=self.channels, kernel_size=3, padding=1)
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvBlocks(in_channels, self.channels, kernel_size=1)
            fpn_conv = ConvBlocks(self.channels, self.channels, kernel_size=3, padding=1)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvBlocks(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1)
        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = _transform_inputs(inputs, in_index=[0,1,2,3], input_transform='multiple_select')

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

if __name__ == '__main__':
    image_size = 512
    in_chans = 3
    backbone = get_convnext(model_name='convnext_base_22k', pretrained=False, in_chans=in_chans)

    img = torch.randn(2, in_chans, image_size, image_size)  # your high resolution picture

    features = backbone(img)
    # print([feature.shape for feature in features], backbone.dims)

    model = UPerHead()
    output = model(features)
    result = resize(
                output,
                size=img.size()[2:],
                mode='bilinear',
                align_corners=False)
    print(result.shape)

