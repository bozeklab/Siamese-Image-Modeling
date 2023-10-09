import os
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

import models_sim
from main_pretrain import DataPreprocessingForSIMWithClasses, DataPreprocessingForSIM
from models_sim import sim_vit_base_patch16_img224
from models_unetr_vit import unetr_vit_small_base_patch16
from models_vit import vit_base_patch16, vit_small_base_patch16
from PIL import Image
import torchvision.transforms.functional as F
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset, ImgWithPickledBoxesDataset

import pickle

from util.pos_embed import interpolate_pos_embed


@dataclass
class Config:
    data_path: str
    model: str
    decoder_embed_dim: int
    input_size: int
    num_boxes: int
    batch_size: int
    init_values: float
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/Downloads/fold_3_256/positive/',
              model='sim_vit_base_patch16_img224',
              input_size=256,
              decoder_embed_dim=768,
              num_boxes=350,
              batch_size=1,
              init_values=None,
              drop_path=0.0)


def prepare_model(chkpt_dir, args):
    # build model
    model = models_sim.__dict__[args.model](norm_pix_loss=False, args=args)

    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    interpolate_pos_embed(model, checkpoint['model'])
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


if __name__ == '__main__':
    transform_inference = DataPreprocessingForSIM(args)
    print(f'Data pre-processing:\n{transform_inference}')

    dataset_inference = ImgWithPickledBoxesDataset(os.path.join(args.data_path), transform=transform_inference,
                                                   )

    print(f'Build dataset: inference images = {len(dataset_inference)}')

    model_sim = prepare_model('checkpoints/sim_base_1600ep_pretrain.pth', args)

    model_sim.eval()

    reps = []
    cls = []
    crops = []

    for idx, sample in tqdm(enumerate(dataset_inference), total=len(dataset_inference)):
        image = sample['x']

        boxes = sample['boxes']
        x = image.unsqueeze(dim=0)
        boxes = boxes.unsqueeze(dim=0)
        box_features = model_sim.forward_boxes(x=x, boxes=boxes)
        reps.append(box_features)

        print(idx)

