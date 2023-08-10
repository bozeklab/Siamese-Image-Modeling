import os
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import RandomSampler
import torch.nn.functional as F

from main_pretrain import DataAugmentationForImagesWithMaps
from models_unetr_vit import cell_vit_base_patch16, unetr_vit_base_patch16

from util.img_with_mask_dataset import PanNukeDataset
from util.post_proc import calculate_instances
from util.tools import calculate_step_metric


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
              batch_size=4,
              init_values=1.0,
              drop_path=0.1)


def compute_metrics(model, sample, num_nuclei_classes):
    x = sample['x']
    tissue_types = sample['tissue_type']

    predictions_ = model(x)

    predictions = model.reshape_model_output(predictions_, 'cpu')

    predictions["tissue_types"] = predictions_["tissue_types"]
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=-1)  # shape: (batch_size, H, W, 2)
    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=-1)  # shape: (batch_size, H, W, num_nuclei_classes)
    predictions["instance_map"], predictions["instance_types"] = model.calculate_instance_map(predictions)  # shape: (batch_size, H', W')
    predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(predictions["instance_map"], predictions["instance_types"])

    # get ground truth values, perform one hot encoding for segmentation maps
    gt_nuclei_binary_map_onehot = (F.one_hot(sample["nuclei_binary_map"], num_classes=2)).type(torch.float32)  # background, nuclei
    nuclei_type_maps = torch.squeeze(sample["nuclei_type_map"]).type(torch.int64)
    gt_nuclei_type_maps_onehot = F.one_hot(nuclei_type_maps, num_classes=num_nuclei_classes).type(torch.float32)


    gt = {
        "nuclei_type_map": gt_nuclei_type_maps_onehot,
        "nuclei_binary_map": gt_nuclei_binary_map_onehot,  # shape: (batch_size, H, W, 2)
        "hv_map": sample["hv_map"],  # shape: (batch_size, H, W, 2)
        "instance_map": sample["instance_map"],  # shape: (batch_size, H, W) -> each instance has one integer
        "instance_types_nuclei": gt_nuclei_type_maps_onehot * sample["instance_map"][..., None],  # shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
        "tissue_types": torch.tensor([PanNukeDataset.tissue_types[t] for t in tissue_types]).type(torch.LongTensor)
    }
    gt["instance_types"] = calculate_instances(gt["nuclei_type_map"], gt["instance_map"])

    batch_metrics, scores = calculate_step_metric(predictions, gt, num_nuclei_classes)
    batch_metrics["tissue_types"] = PanNukeDataset.tissue_types.keys()

    return batch_metrics, scores


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
    num_nuclei_classes = len(PanNukeDataset.nuclei_types)
    num_tissue_classes = len(PanNukeDataset.tissue_types)

    model_cell_vit = prepare_model('checkpoints/checkpoint-latest.pth',
                                   init_values=args.init_values,
                                   drop_path_rate=args.drop_path,
                                   num_nuclei_classes=num_nuclei_classes,
                                   num_tissue_classes=num_tissue_classes,
                                   embed_dim=args.embed_dim,
                                   extract_layers=args.extract_layers)

    model_cell_vit.eval()

    transform_maps = DataAugmentationForImagesWithMaps(args)

    dataset_inference = PanNukeDataset(os.path.join(args.data_path), transform=transform_maps)
    print(f'Build dataset: inference images = {len(dataset_inference)}')

    sampler_inference = RandomSampler(dataset_inference)
    dataloader_inference = torch.utils.data.DataLoader(
        dataset_inference, sampler=sampler_inference,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    for idx, sample in enumerate(dataloader_inference):

        x = sample['x']
        type_map = sample['nuclei_type_map']
        inst_map = sample['instance_map']
        hv_map = sample['hv_map']
        tissue_type = sample['tissue_type']

        #z = model_cell_vit(x)
        batch_metrics, scores = compute_metrics(model_cell_vit, sample, num_nuclei_classes)

        if idx == 5:
            break