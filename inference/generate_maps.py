import os
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm

from main_pretrain import DataPreprocessingForSIM, DataAugmentationForImagesWithMaps
from models_unetr_vit import cell_vit_base_patch16, unetr_vit_base_patch16

from util.img_with_mask_dataset import ImagesWithSegmentationMaps
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset

import pickle

tissue_types = {"Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, "Cervix": 4,
                "Colon": 5, "Esophagus": 6, "HeadNeck": 7, "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
                "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15, "Testis": 16, "Thyroid": 17, "Uterus": 18}

nuclei_types = {"Background": 0, "Neoplastic": 1, "Inflammatory": 2, "Connective": 3,
                "Dead": 4, "Epithelial": 5}


@dataclass
class Config:
    data_path: str
    input_size: int
    batch_size: int
    embed_dim: int
    extract_layers: List[int]
    init_values: float
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/data/pannuke/fold1',
              input_size=352,
              embed_dim=768,
              extract_layers=[3, 6, 9, 12],
              batch_size=1,
              init_values=1.0,
              drop_path=0.1)


def prepare_model(chkpt_dir_vit, **kwargs):
    # build ViT encoder

    num_nuclei_classes = kwargs.pop('num_nuclei_classes')
    num_tissue_classes = kwargs.pop('num_tissue_classes')
    embed_dim = kwargs.pop('embed_dim')
    extract_layers = kwargs.pop('extract_layers')
    drop_rate = kwargs['drop_path_rate']

    vit_encoder = unetr_vit_base_patch16(num_classes=num_tissue_classes, **kwargs)

    # load ViT model
    checkpoint = torch.load(chkpt_dir_vit, map_location='cpu')
    msg = vit_encoder.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    model = cell_vit_base_patch16(num_nuclei_classes=num_nuclei_classes,
                                  embed_dim=embed_dim,
                                  extract_layers=extract_layers,
                                  drop_rate=drop_rate,
                                  encoder=vit_encoder)

    # load model
    return model


if __name__ == '__main__':

    num_nuclei_classes = len(nuclei_types)
    num_tissue_classes = len(tissue_types)

    model_cell_vit = prepare_model('checkpoints/checkpoint-latest.pth',
                                   init_values=args.init_values,
                                   drop_path_rate=args.drop_path,
                                   num_nuclei_classes=num_nuclei_classes,
                                   num_tissue_classes=num_tissue_classes,
                                   embed_dim=args.embed_dim,
                                   extract_layers=args.extract_layers)

    model_cell_vit.eval()

    transform_maps = DataAugmentationForImagesWithMaps(args)

    dataset_inference = ImagesWithSegmentationMaps(os.path.join(args.data_path), transform=transform_maps)

    for idx, sample in enumerate(dataset_inference):
        x = sample['x']
        type_map = sample['type_map']
        inst_map = sample['inst_map']
        hv_map = sample['hv_map']

        x = x.unsqueeze(dim=0)
        z = model_cell_vit(x)

        print(z.keys())

        nuclei_type_map = z['nuclei_type_map']
        nuclei_type_map.squeeze()
        print(nuclei_type_map)

        if idx == 5:
            break