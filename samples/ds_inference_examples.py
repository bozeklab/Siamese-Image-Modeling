import os
from dataclasses import dataclass

import torch
from torch.utils.data import RandomSampler
from torchvision import transforms

from main_pretrain import DataPreprocessingForSIMWithClasses
from samples.ds_pretrain_example import tensor_batch_to_list, draw_bboxes, create_image_grid
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset


@dataclass
class Config:
    data_path: str
    input_size: int
    num_boxes: int
    batch_size: int


args = Config(data_path='/Users/piotrwojcik/data/he/positive', input_size=352, num_boxes=250, batch_size=2)


if __name__ == '__main__':
    transform_inference = DataPreprocessingForSIMWithClasses(args)
    print(f'Data pre-processing:\n{transform_inference}')

    dataset_inference = ImgWithPickledBoxesAndClassesDataset(os.path.join(args.data_path), transform=transform_inference)

    print(f'Build dataset: inference images = {len(dataset_inference)}')

    sampler_train = RandomSampler(dataset_inference)
    dataloader_inference = torch.utils.data.DataLoader(
        dataset_inference, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    images = []
    for idx, sample in enumerate(dataloader_inference):
        x = sample['x']
        boxes = sample['boxes']
        classes = sample['classes']

        x = x.permute(0, 2, 3, 1)

        x = tensor_batch_to_list(x)
        x = draw_bboxes(x, boxes=boxes)
        images.extend(x)

        if idx == 1:
            break

    create_image_grid(images, num_cols=2)
