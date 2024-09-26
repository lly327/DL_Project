from torch.utils.data import Dataset
import glob
import cv2
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import random
import os


class BASEDATASETS_VOC(Dataset):
    def __init__(self, data_root, pipeline, resize_shape=[512,512], is_training=True):
        self.data_root = data_root
        self.image_root = self.data_root + 'JPEGImages/'
        self.png_root = self.data_root + 'SegmentationClass/'
        self.resize_shape = resize_shape
        self.is_training = is_training

        self.image_path_list = glob.glob(self.image_root + '*.jpg')
        self.mask_path_list = glob.glob(self.png_root + '*.png')
        assert len(self.image_path_list) == len(self.mask_path_list), 'The number of images and masks should be the same.'
        train_txt_path = os.path.join(data_root, 'train.txt')
        val_txt_path = os.path.join(data_root, 'val.txt')
        assert os.path.exists(train_txt_path) and os.path.exists(val_txt_path), 'train.txt or val.txt not exists.'
        if self.is_training:
            with open(train_txt_path, 'r') as f:
                lines = f.read().splitlines()
            self.image_path_list = [os.path.join(self.image_root, line+'.jpg') for line in lines]
            self.mask_path_list = [os.path.join(self.png_root, line+'.png') for line in lines]
        else:
            with open(val_txt_path, 'r') as f:
                lines = f.read().splitlines()
            self.image_path_list = [os.path.join(self.image_root, line+'.jpg') for line in lines]
            self.mask_path_list = [os.path.join(self.png_root, line+'.png') for line in lines]

        # self.ann_file = self.data_root + 'ImageSets/Main/trainval.txt'
        # self.classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        #                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        #                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        # self.num_classes = len(self.classes)
        # self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        # self.ind_to_class = dict(zip(range(self.num_classes), self.classes))

        self.pipeline = pipeline

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def load_annotations(self):
        pass

    def prepare_data(self, idx):
        image_path = self.image_path_list[idx]
        mask_path = self.mask_path_list[idx]

        # 打开 PNG 图片
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # 获取原始尺寸
        original_width, original_height = image.size
        base_size = self.resize_shape[0]
        # 计算比例
        if original_width > original_height:
            # 如果宽度大于高度，按宽度为 512 进行缩放
            new_width = base_size
            new_height = int((base_size / original_width) * original_height)
        else:
            # 如果高度大于宽度，按高度为 512 进行缩放
            new_height = base_size
            new_width = int((base_size / original_height) * original_width)

        # 按比例缩放
        resized_image = image.resize((new_width, new_height))
        resized_mask = mask.resize((new_width, new_height))

        # new_image = Image.new("RGB", (base_size, base_size), (0, 0, 0))
        # new_mask = Image.new("L", (base_size, base_size), (0))
        new_image = np.zeros((base_size, base_size, 3), dtype=np.uint8)
        new_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        
        # 计算缩放后图片在新画布上的位置，使其居中
        top_left_x = (base_size - new_width) // 2
        top_left_y = (base_size - new_height) // 2

        # 将缩放后的图片粘贴到黑色画布上
        # new_image.paste(resized_image, (top_left_x, top_left_y))
        # new_mask.paste(resized_mask, (top_left_x, top_left_y))
        new_image[top_left_y:top_left_y+resized_image.size[1], top_left_x:top_left_x+resized_image.size[0]] = np.array(resized_image)
        new_mask[top_left_y:top_left_y+resized_image.size[1], top_left_x:top_left_x+resized_image.size[0]] = np.array(resized_mask)
            
        if self.pipeline is not None:
            image = self.pipeline(new_image)
        
        mask = torch.tensor(np.array(new_mask)).unsqueeze(0).float()

        result_dict = {
            'image': image,
            'mask': mask,
            'original_width': original_width,
            'original_height': original_height
        }
        
        return result_dict
        # image.palette.colors
        


if __name__ == '__main__':
    data_root = '/Users/lly/codes/datasets/Pascal_voc_test_data/data_dataset_voc/'
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets = BASEDATASETS_VOC(data_root=data_root, pipeline=pipeline, is_training=True)
    print(len(datasets))
    dataloader = DataLoader(datasets, batch_size=2, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        print(data['image'].shape, data['mask'].shape)
        break
