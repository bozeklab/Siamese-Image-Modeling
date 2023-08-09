import os
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from main_pretrain import DataPreprocessingForSIM
from models_vit import vit_base_patch16
from PIL import Image
import torchvision.transforms.functional as F
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset

import pickle

@dataclass
class Config:
    data_path: str
    input_size: int
    batch_size: int
    init_values: float
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/data/he/positive',
              input_size=352,
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
