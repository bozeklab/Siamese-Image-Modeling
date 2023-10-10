import os
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

import models_sim
from main_pretrain import DataPreprocessingForSIMWithClasses, DataPreprocessingForSIM, DataAugmentationForSIMTraining
from models_sim import sim_vit_base_patch16_img224
from models_unetr_vit import unetr_vit_small_base_patch16
from models_vit import vit_base_patch16, vit_small_base_patch16
from PIL import Image
import torchvision.transforms.functional as F

from util.datasets import ImagenetWithMask
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
    use_abs_pos_emb: bool
    drop_path_rate: float
    init_values: float
    loss_type: str
    projector_depth: int
    predictor_depth: int
    use_proj_ln: bool
    use_pred_ln: bool
    online_ln: bool
    train_patch_embed: bool
    with_blockwise_mask: bool
    crop_min: float
    blockwise_num_masking_patches: int
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/Downloads/fold_3_256/positive/',
              model='sim_vit_base_patch16_img224',
              input_size=224,
              blockwise_num_masking_patches=127,
              crop_min=0.2,
              num_boxes=150,
              decoder_embed_dim=768,
              with_blockwise_mask=True,
              drop_path_rate=0.0,
              predictor_depth=4,
              projector_depth=2,
              use_proj_ln=False,
              use_pred_ln=False,
              train_patch_embed=False,
              online_ln=False,
              batch_size=1,
              loss_type='sim',
              use_abs_pos_emb=True,
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
    transform_train = DataAugmentationForSIMTraining(args)
    print(f'Data pre-processing:\n{transform_train}')

    dataset_train = ImagenetWithMask(os.path.join(args.data_path),
                                     input_size=args.input_size,
                                     transform=transform_train,
                                     with_blockwise_mask=args.with_blockwise_mask,
                                     blockwise_num_masking_patches=args.blockwise_num_masking_patches)

    print(f'Build dataset: inference images = {len(dataset_train)}')

    model_sim = prepare_model('checkpoints/sim_base_1600ep_pretrain.pth', args)

    model_sim.eval()

    reps = []

    for idx, sample in tqdm(enumerate(dataset_train), total=len(dataset_train)):
        samples, mask = sample
        x0 = samples['x0']
        x1 = samples['x1']
        x2 = samples['x2']
        i1, i2 = samples['i1'], samples['i2']
        j1, j2 = samples['j1'], samples['j2']
        h1, h2 = samples['h1'], samples['h2']
        w1, w2 = samples['w1'], samples['w2']
        boxes0 = samples['boxes0']
        boxes1 = samples['boxes1']
        boxes2 = samples['boxes2']

        #x = image.unsqueeze(dim=0)
        #boxes = boxes.unsqueeze(dim=0)
        #repr = model_sim(x)
        reps.append(repr)


