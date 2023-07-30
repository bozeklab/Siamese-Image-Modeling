import argparse
import os
from dataclasses import dataclass

import torch
from torch.utils.data import SequentialSampler

from main_pretrain import DataPreprocessingForSIM
from models_vit import vit_base_patch16
from util.img_with_picke_dataset import ImgWithPickledBoxesAndClassesDataset


@dataclass
class Config:
    data_path: str
    input_size: int
    num_boxes: int
    batch_size: int
    init_values: float
    drop_path: float


args = Config(data_path='/data/pwojcik/he/positive',
              input_size=352,
              num_boxes=250,
              batch_size=1,
              init_values=1.0,
              drop_path=0.1)


def prepare_model(chkpt_dir, **kwargs):
    # build model
    model = vit_base_patch16(**kwargs)

    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


if __name__ == '__main__':
    #args = get_args_parser()
    #args = args.parse_args()

    transform_inference = DataPreprocessingForSIM(args)
    print(f'Data pre-processing:\n{transform_inference}')

    dataset_inference = ImgWithPickledBoxesAndClassesDataset(os.path.join(args.data_path), transform=transform_inference)

    print(f'Build dataset: inference images = {len(dataset_inference)}')

    model_sim = prepare_model('/data/pwojcik/SimMIM/output_dir/checkpoint-latest.pth',
                              init_values=args.init_values,
                              global_pool=True,
                              drop_path_rate=args.drop_path,
                              box_patch_size=8)

    for idx, sample in enumerate(dataset_inference):
        x = sample['x']
        boxes = sample['boxes']
        x = x.unsqueeze(dim=0)
        boxes = boxes.unsqueeze(dim=0)
        box_features = model_sim.forward_boxes(x=x, boxes=boxes)
        print(box_features.shape)
