import os
from dataclasses import dataclass

import torch
from torch.utils.data import RandomSampler
from torchvision import transforms

from main_pretrain import DataPreprocessingForSIM, DataAugmentationForImagesWithMasks
from test_examples.ds_pretrain_example import tensor_batch_to_list, draw_bboxes, create_image_grid
from util.img_with_mask_dataset import ImagesWithSegmentationMasks


@dataclass
class Config:
    data_path: str
    input_size: int
    batch_size: int


args = Config(data_path='/Users/piotrwojcik/data/pannuke/fold1', input_size=352, batch_size=2)


if __name__ == '__main__':
    transform_mask = DataAugmentationForImagesWithMasks(args)

    print(f'Segmentation data transform:\n{transform_mask}')

    mask_train = ImagesWithSegmentationMasks(root=os.path.join(args.data_path),
                                             transform=transform_mask)

    print(f'Build dataset: images with mask = {len(mask_train)}')

    for idx, data in enumerate(mask_train):
        x = data['x']
        imap = data['inst_map']
        tmap = data['type_map']

        if idx == 4:
            break

