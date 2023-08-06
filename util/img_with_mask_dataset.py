import os

import torch
import torchvision.datasets as datasets
import imageio
import pickle
import numpy as np


class ImagesWithSegmentationMasks(datasets.VisionDataset):
    def __init__(self, root, transform):
        super(ImagesWithSegmentationMasks, self).__init__(root, transform=transform)

        self.root_imgs = os.path.join(root, 'images')
        self.root_masks = os.path.join(root, 'masks')
        self.transform = transform

        self.file_list = [filename for filename in os.listdir(self.root_imgs) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root_imgs, filename)
        mask_path = os.path.join(self.root_masks, os.path.splitext(filename)[0] + '.npy')

        image = imageio.v3.imread(image_path)
        seg = np.load(mask_path, allow_pickle=True).item()
        type_map = seg['type_map']
        inst_map = seg['instance_map']

        return self.transform(image, type_map, inst_map)

    def __len__(self):
        return len(self.file_list)