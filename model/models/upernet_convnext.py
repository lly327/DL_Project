import torch
import numpy as np
import torch.nn as nn
from model.backbone.convnext import get_convnext
from model.head.upernet import UPerHead
from model.head.fcn_head import FCNHead
import torch.nn.functional as F


class UpernetConvnext(nn.Module):
    def __init__(self, model_name='convnext_base_22k', pretrained=False, in_chans=3,
                 num_classes=2, use_aux_head=True):
        super().__init__()
        """
            model_name: backbone的名称,default='convnext_base_22k',可选项['convnext'_'tiny|small|base|large|xlarge'_'1k|22k']
            pretrained: 是否使用预训练模型,default=False
            in_chans: 输入图片的通道数,default=3
            num_classes: 分类的类别数,default=2
        """
        self.backbone = get_convnext(model_name=model_name, pretrained=pretrained, in_chans=in_chans)
        self.head = UPerHead(num_classes=num_classes)
        self.use_aux_head = use_aux_head
        if use_aux_head:
            self.aux_head = FCNHead(in_channels=512, mid_channels=256, num_classes=num_classes, num_convs=1, concat_input=False)

        
    def forward(self, data):
        x = data['image']
        features = self.backbone(x)
        output = self.head(features)
        if self.use_aux_head:
            aux_head_output = self.aux_head(features[2])
        
        output_shape = data['resized_shape']
        output_shape = int(output_shape[0][0]), int(output_shape[0][0])

        result = F.interpolate(output, size=output_shape, scale_factor=None, mode='bilinear', align_corners=None)
        if self.use_aux_head:
            aux_head_result = F.interpolate(aux_head_output, size=output_shape, scale_factor=None, mode='bilinear', align_corners=None)
            return result, aux_head_result
        else:
            return result