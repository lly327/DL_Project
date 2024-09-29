import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from model.backbone.convnext import get_convnext
from model.head.upernet import UPerHead
from model.models.upernet_convnext import UpernetConvnext
from data.trans import *
from utils.utils import _expand_onehot_labels_dice

from data.BaseDatasets import BASEDATASETS_VOC
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler
from model.head.upernet import resize
import torch.nn.functional as F
import tqdm
from sklearn.metrics import f1_score
import shutil
import cv2

def train_epoch(epoch, model, train_dataloader, loss_function, optimizer, use_aux_head):
    model.train()
    train_dataloader = tqdm.tqdm(train_dataloader)
    for i, data in enumerate(train_dataloader):
        mask = data['mask']
        output = model(data)
        if use_aux_head:
            loss1 = loss_function(output[0], mask.squeeze(1))
            loss2 = loss_function(output[1], mask.squeeze(1))
            loss = loss1 + 0.4 * loss2
        else:
            loss = loss_function(output, mask.squeeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def val_epoch(epoch, model, val_dataloader, loss_function, optimizer, num_classes, output_dir, seg_name, output_txt_name, use_aux_head):
    all_f1 = [[] for _ in range(num_classes)]
    losses = []
    model.eval()
    val_dataloader = tqdm.tqdm(val_dataloader)
    for i, data in enumerate(val_dataloader):
        mask = data['mask']
        with torch.no_grad():
            output = model(data)
            if use_aux_head:
                loss1 = loss_function(output[0], mask.squeeze(1))
                loss2 = loss_function(output[1], mask.squeeze(1))
                loss = loss1 + 0.4 * loss2
                output_squeeze = output[0].squeeze(0)
            else:
                loss = loss_function(output, mask.squeeze(1))
                output_squeeze = output.squeeze(0)
            losses.append(loss.item())

            mask_squeeze = mask.squeeze(0).squeeze(0)
            one_hot_target = _expand_onehot_labels_dice(output_squeeze, mask_squeeze)
            one_hot_target = one_hot_target.cpu().numpy()

            # 对pred进行处理
            # 找到每个像素点最大值所在的通道位置
            max_channel_index = torch.argmax(output_squeeze, dim=0)
            one_hot_pred = _expand_onehot_labels_dice(output_squeeze, max_channel_index)
            one_hot_pred = one_hot_pred.cpu().numpy()

            for i in range(num_classes):
                # 计算F1分数，平均方法可以是'micro', 'macro', 'weighted'
                f1 = f1_score(one_hot_target[i], one_hot_pred[i], average='micro', zero_division=1.0)
                all_f1[i].append(f1)
    num_images = len(all_f1[0])
    f1_mean_value = []
    for i in range(len(seg_name)):
        f1_mean_value.append(np.sum(all_f1[i]) / num_images)
    with open(os.path.join(output_dir, output_txt_name), 'a') as f:
        f.writelines(f"Epoch:{epoch}\n")
        for i in range(len(seg_name)):
            f.writelines(f"{seg_name[i]} F1 value:{f1_mean_value[i]}\n")
        f.writelines(f"Loss value:{np.sum(losses) / num_images}\n")
        f.writelines("\n")
    print(f"Epoch: {epoch}, ValLoss: {np.sum(losses) / num_images}", end=' ')
    for i in range(len(seg_name)):
        print(f"{seg_name[i]} F1 value:{f1_mean_value[i]}", end=' ')
    print()

    return np.array(f1_mean_value).mean()


def save_image_epoch(epoch, model, val_dataloader, num_classes, output_dir, seg_name, use_aux_head):
    """保存模型的预测结果"""
    model.eval()
    val_dataloader = tqdm.tqdm(val_dataloader)
    color_list = [[0,0,0], [255,255,255], [0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255], [255,255,0]]
    for i, data in enumerate(val_dataloader):
        image_path = data['image_path'][0]
        new_h_w = [int(data['new_h_w'][0]), int(data['new_h_w'][1])]
        top_left_x_y = [int(data['top_left'][0]), int(data['top_left'][1])]
        ori_w, roi_h = int(data['original_width'][0]), int(data['original_height'][0])
        with torch.no_grad():
            if use_aux_head:
                output = model(data)[0].squeeze(0)
            else:
                output = model(data).squeeze(0)

            arg_result = np.argmax(output.cpu().numpy(), axis=0)
            mask_result = np.zeros((arg_result.shape[0], arg_result.shape[1], 3), dtype=np.uint8)
            for i in range(num_classes):
                mask_result[arg_result == i, ] = color_list[i]

            # 对mask_result先裁剪 再reshape， mmseg使用的方法
            mask_result_crop = mask_result[top_left_x_y[1]:top_left_x_y[1]+new_h_w[0], top_left_x_y[0]:top_left_x_y[0]+new_h_w[1], ]
            mask_result_resize = cv2.resize(mask_result_crop, (ori_w, roi_h)).astype(np.uint8)

            ori_img = cv2.imread(image_path)
            final_reuslt = cv2.addWeighted(ori_img, 0.5, mask_result_resize, 0.5, 0)

            cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), final_reuslt)



if __name__ == "__main__":
    # 全局设置
    is_cuda = False
    output_dir = '/Users/lly/codes/DL_Project/output'
    train_bs = 2
    val_bs = 1
    save_epoch_times = 1
    val_epoch_times = 1
    save_image_times = 1
    whole_epoch = 200

    # 数据集设置
    seg_name = ['backgroud', 'cat']
    data_root = '/Users/lly/codes/datasets/Pascal_voc_test_data/data_dataset_voc/'
    pipeline_type = 'albu'
    use_PhotoMetricDistortion = True
    dataset_dict = {
        'jpg_dir_name': 'JPEGImages',
        'png_dir_name': 'SegmentationClass',
        'train_txt_name': 'train.txt',
        'val_txt_name': 'val.txt'
    }

    # 模型设置
    model_name = 'convnext_base_22k'
    backbone_pretrain = True
    in_chans = 3
    num_classes = len(seg_name)
    checkpoint_path = None
    use_axu_head = True

    # 损失函数设置

    # 优化器设置

    # 输出设置
    output_txt_name = 'val_metric.txt'

    ######################################################  分隔符

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(output_dir, 'config_train.py'))

    model = UpernetConvnext(model_name=model_name, pretrained=backbone_pretrain, in_chans=in_chans, 
                            num_classes=num_classes, use_aux_head=use_axu_head)
    if checkpoint_path:
        model = torch.load(checkpoint_path, map_location='cpu')
        # if 'state_dict' in sd.keys():
        #     sd = sd['state_dict']
        # model.load_state_dict(sd)
        print(f"Load model from {checkpoint_path}")
    
    assert pipeline_type in ['albu', 'norm'], 'pipeline_type should be albu or norm'
    if pipeline_type == 'norm':
        train_pipeline = get_norm_transforms()
        val_pipeline = get_norm_transforms()
    elif pipeline_type == 'albu':
        train_pipeline = get_Albu_transforms('train')
        val_pipeline = get_Albu_transforms('val')
    
    train_datasets = BASEDATASETS_VOC(data_root=data_root, pipeline=train_pipeline, is_training=True, is_cuda=is_cuda, use_PhotoMetricDistortion=use_PhotoMetricDistortion, **dataset_dict)
    train_dataloader = DataLoader(dataset=train_datasets, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)
    val_datasets = BASEDATASETS_VOC(data_root=data_root, pipeline=train_pipeline, is_training=False, is_cuda=is_cuda, use_PhotoMetricDistortion=use_PhotoMetricDistortion, **dataset_dict)
    val_dataloader = DataLoader(dataset=val_datasets, batch_size=val_bs, shuffle=True, num_workers=0, drop_last=False)

    with open(os.path.join(output_dir, output_txt_name), 'a') as f:
        f.writelines(f"val images num: {len(val_dataloader)}\n")
    
    loss_function = torch.nn.CrossEntropyLoss()

    if is_cuda:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        loss_function = loss_function.cuda()
    
    parameters = model.parameters()
    opetimizer = optim.Adam(parameters, lr=0.0001, betas=(0.9,0.999), weight_decay=0.05)
    scheduler = lr_scheduler.ReduceLROnPlateau(opetimizer, 'min', factor=0.6, patience=5)

    max_f1 = 0
    for epoch in range(1, whole_epoch):
        train_epoch(epoch, model, train_dataloader, loss_function, opetimizer, use_axu_head)

        if epoch % val_epoch_times == 0:
            mean_f1 = val_epoch(epoch, model, val_dataloader, loss_function, opetimizer, num_classes, 
                                output_dir, seg_name, output_txt_name, use_axu_head)
        
        if epoch % save_epoch_times == 0 or mean_f1 > max_f1:
            if mean_f1 > max_f1:
                max_f1 = mean_f1
            print("Save checkpoint")
            torch.save(model, f'{output_dir}/model_{str(epoch)}_{round(mean_f1, 3)}.pth')
        
        if epoch % save_image_times == 0:
            image_output_dir = os.path.join(output_dir, f'val_result_{str(epoch)}')
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir, exist_ok=True)
            save_image_epoch(epoch, model, val_dataloader, num_classes, image_output_dir, seg_name, use_axu_head)

    print("Train Done!")

    pass
