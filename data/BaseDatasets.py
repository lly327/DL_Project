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
from data.trans import PhotoMetricDistortion


class BASEDATASETS_VOC(Dataset):
    def __init__(self, data_root, pipeline, resize_shape=[512,512], is_training=True, is_cuda=False, use_PhotoMetricDistortion=False, **cfg):
        self.data_root = data_root
        self.image_root = self.data_root + 'JPEGImages/'
        self.png_root = self.data_root + 'SegmentationClass/'
        self.resize_shape = resize_shape
        self.is_training = is_training
        self.is_cuda = is_cuda
        self.use_PhotoMetricDistortion = use_PhotoMetricDistortion

        self.image_path_list = glob.glob(self.image_root + '*.jpg')
        self.mask_path_list = glob.glob(self.png_root + '*.png')
        assert len(self.image_path_list) == len(self.mask_path_list), 'The number of images and masks should be the same.'
        train_txt_path = os.path.join(data_root, cfg['train_txt_name'])
        val_txt_path = os.path.join(data_root, cfg['val_txt_name'])
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
        
        if self.use_PhotoMetricDistortion:
            self.photometric_trans = PhotoMetricDistortion()
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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.pipeline is not None:
            if 'albu' in self.pipeline.__module__:
                image = self.pipeline(image=image)['image']
            else:
                image = self.pipeline(image)
        if self.use_PhotoMetricDistortion:
            image = self.photometric_trans.transform(image)

        # 打开 PNG 图片
        mask = Image.open(mask_path)
        # 获取原始尺寸
        original_width, original_height = mask.size
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
        resized_image = cv2.resize(image, (new_width, new_height))/255.0
        resized_image = (resized_image-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        resized_mask = mask.resize((new_width, new_height))

        new_image = np.zeros((base_size, base_size, 3))
        new_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        
        # 计算缩放后图片在新画布上的位置，使其居中
        top_left_x = (base_size - new_width) // 2
        top_left_y = (base_size - new_height) // 2

        # 将缩放后的图片粘贴到黑色画布上
        new_image[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = np.array(resized_image)
        new_mask[top_left_y:top_left_y+new_height, top_left_x:top_left_x+new_width] = np.array(resized_mask)
        
        image = torch.tensor(np.array(new_image)).permute(2, 0, 1).float()
        mask = torch.tensor(np.array(new_mask)).unsqueeze(0).long()

        if self.is_cuda:
            image = image.cuda()
            mask = mask.cuda()

        result_dict = {
            'image': image,
            'mask': mask,
            'resized_shape': self.resize_shape,
            'original_width': original_width,
            'original_height': original_height,
            'image_path': image_path,
            'mask_path': mask_path,
            'new_h_w': [new_height, new_width],
            'top_left': [top_left_x, top_left_y]
        }
        
        return result_dict
        # image.palette.colors

class TEST_DATASETS(Dataset):
    def __init__(self, data_root, pipeline, resize_shape=[512,512], is_cuda=False, **cfg) -> None:
        super().__init__()


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
