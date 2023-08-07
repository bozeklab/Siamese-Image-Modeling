import os
from dataclasses import dataclass

import imageio
import imgaug as ia
import torch
from torch.utils.data import RandomSampler
from torchvision import transforms

from main_pretrain import DataPreprocessingForSIM, DataAugmentationForImagesWithMaps
from test_examples.ds_pretrain_example import tensor_batch_to_list, draw_bboxes, create_image_grid
from util.img_with_mask_dataset import ImagesWithSegmentationMaps


@dataclass
class Config:
    data_path: str
    input_size: int
    batch_size: int


args = Config(data_path='/Users/piotrwojcik/data/pannuke/fold1', input_size=352, batch_size=2)


if __name__ == '__main__':
    transform_mask = DataAugmentationForImagesWithMaps(args)

    print(f'Segmentation data transform:\n{transform_mask}')

    mask_train = ImagesWithSegmentationMaps(root=os.path.join(args.data_path),
                                            transform=transform_mask)

    print(f'Build dataset: images with mask = {len(mask_train)}')

    cells = []

    idxs = [9, 10, 12,  34]

    for idx, data in enumerate(mask_train):
        x0 = data['x0']
        x = data['x']
        imap = data['inst_map']
        tmap = data['type_map']

        if idx == 40:
            break
        if not idx in idxs:
            continue

        cells.append(x0)  # column 1
        cells.append(x)  # column 2
        cells.append(imap.draw_on_image(x)[0])  # column 3
        cells.append(tmap.draw_on_image(x)[0])  # column 4

        cells.append(imap.draw(size=x.shape[:2])[0])    # column 5
        cells.append(tmap.draw(size=x.shape[:2])[0])    # column 6

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=6)
    imageio.imwrite("ds_segmaps.jpg", grid_image)