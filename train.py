import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from model.backbone.convnext import get_convnext
from model.head.upernet import UPerHead
from data.BaseDatasets import BASEDATASETS_VOC
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler
from model.head.upernet import resize
import torch.nn.functional as F

def train_epoch():
    pass

def val_epoch():
    pass


if __name__ == "__main__":
    backbone = get_convnext(model_name='convnext_base_22k', pretrained=False, in_chans=3)
    model = UPerHead(num_classes=1)

    data_root = '/Users/lly/codes/datasets/Pascal_voc_test_data/data_dataset_voc/'
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets = BASEDATASETS_VOC(data_root=data_root, pipeline=pipeline, is_training=True)
    train_dataloader = DataLoader(dataset=datasets, batch_size=2, shuffle=True, num_workers=0)

    loss_function = torch.nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5)  # 2->5

    whole_epoch = 100
    for epoch in range(whole_epoch):
        for i, data in enumerate(train_dataloader):
            image = data['image']
            mask = data['mask']
            output = model(backbone(image))

            output = resize(
                output,
                size=mask.size()[2:],
                mode='bilinear',
                align_corners=False)

            loss = loss_function(output, mask)

            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pass
