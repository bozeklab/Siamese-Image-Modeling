import os
from dataclasses import dataclass

import imageio
import random
import imgaug as ia
import torch
import matplotlib.pyplot as plt
from imgaug import SegmentationMapsOnImage
from torch.utils.data import RandomSampler
from torchvision import transforms

from main_pretrain import DataPreprocessingForSIM, DataAugmentationForImagesWithMaps
from samples.ds_pretrain_example import tensor_batch_to_list, draw_bboxes, create_image_grid
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

    idxs = []
    for _ in range(4):
        num = random.randint(1, 40)  # Generate a random number between 1 and 99 (both inclusive)
        idxs.append(num)

    for idx, data in enumerate(mask_train):
        x0 = data['x0']
        x = data['x']
        imap = data['instance_map']
        tmap = data['nuclei_type_map']
        hvmap = data['hv_map']

        if idx == 40:
            break
        if not idx in idxs:
            continue

        x0 = torch.einsum('chw -> hwc', x0)
        x0 = (x0 * 255.0).to(torch.uint8).numpy()

        x = torch.einsum('chw -> hwc', x)
        x = (x * 255.0).to(torch.uint8).numpy()

        cells.append(x0)  # column 1
        cells.append(x)  # column 2

        imap = SegmentationMapsOnImage(imap.numpy(), shape=x.shape)
        tmap = SegmentationMapsOnImage(tmap.numpy(), shape=x.shape)

        cells.append(imap.draw_on_image(x)[0])  # column 3
        cells.append(tmap.draw_on_image(x)[0])  # column 4

        cells.append(imap.draw(size=x.shape[:2])[0])    # column 5
        cells.append(tmap.draw(size=x.shape[:2])[0])    # column 6

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=6)
    imageio.imwrite("ds_segmaps.jpg", grid_image)