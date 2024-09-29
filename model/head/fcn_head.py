# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from model.base_modules import ConvBlocks


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels=512,
                 mid_channels=256,
                 num_classes=6,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 dropout_ratio=0.1,
                 **kwargs):
        super().__init__()
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.in_channels = in_channels
        self.channels = mid_channels
        self.out_channels = num_classes
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvBlocks(self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation)
        )
        for i in range(num_convs - 1):
            convs.append(
                ConvBlocks(self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation)
            )
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvBlocks(self.in_channels+self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size//2, dilation=dilation)
            
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def _forward_feature(self, x):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def cls_seg(self, feat):
        "Classify each pixel."
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
