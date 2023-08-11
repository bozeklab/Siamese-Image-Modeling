import os
from dataclasses import dataclass
from typing import List

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import RandomSampler
import torch.nn.functional as F
from tqdm import tqdm

from main_pretrain import DataAugmentationForImagesWithMaps
from models_unetr_vit import cell_vit_base_patch16, unetr_vit_base_patch16

from util.img_with_mask_dataset import PanNukeDataset
from util.metrics import cell_detection_scores
from util.post_proc import calculate_instances
from util.tools import calculate_step_metric


@dataclass
class Config:
    data_path: str
    device: str
    input_size: int
    batch_size: int
    embed_dim: int
    extract_layers: List[int]
    init_values: float
    drop_path: float


args = Config(data_path='/data/pwojcik/SimMIM/pannuke/fold1/',
              input_size=352,
              embed_dim=768,
              extract_layers=[3, 6, 9, 12],
              batch_size=4,
              init_values=1.0,
              drop_path=0.1,
              device='gpu')


def compute_metrics(model, sample, num_nuclei_classes):
    x = sample['x']
    nuclei_type_map = sample['nuclei_type_map']
    instance_map = sample['instance_map']
    nuclei_binary_map = sample['nuclei_binary_map']
    hv_map = sample['hv_map']
    tissue_types = sample['tissue_type']

    x = x.to(device, non_blocking=True)
    nuclei_type_map = nuclei_type_map.to(device, non_blocking=True)
    instance_map = instance_map.to(device, non_blocking=True)
    nuclei_binary_map = nuclei_binary_map.to(device, non_blocking=True)
    hv_map = hv_map.to(device, non_blocking=True)
    tissue_types = tissue_types.to(device, non_blocking=True)

    predictions_ = model(x)
    predictions = model.reshape_model_output(predictions_, x.device)

    predictions["tissue_types"] = predictions_["tissue_types"]
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=-1)  # (batch_size, H, W, 2)
    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=-1)  # (batch_size, H, W, num_nuclei_classes)
    predictions["instance_map"], predictions["instance_types"] = model.calculate_instance_map(predictions)  # (batch_size, H', W')
    predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(predictions["instance_map"], predictions["instance_types"])

    # get ground truth values, perform one hot encoding for segmentation maps
    gt_nuclei_binary_map_onehot = (F.one_hot(nuclei_binary_map, num_classes=2)).type(torch.float32)  # background, nuclei
    nuclei_type_maps = torch.squeeze(nuclei_type_map).type(torch.int64)
    gt_nuclei_type_maps_onehot = F.one_hot(nuclei_type_maps, num_classes=num_nuclei_classes).type(torch.float32)

    gt = {
        "nuclei_type_map": gt_nuclei_type_maps_onehot,
        "nuclei_binary_map": gt_nuclei_binary_map_onehot,  # (batch_size, H, W, 2)
        "hv_map": hv_map,  # (batch_size, H, W, 2)
        "instance_map": instance_map,  # (batch_size, H, W) -> each instance has one integer
        "instance_types_nuclei": gt_nuclei_type_maps_onehot * instance_map[..., None],  # (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
        "tissue_types": torch.tensor([PanNukeDataset.tissue_types[t] for t in tissue_types]).type(torch.LongTensor)
    }
    gt["instance_types"] = calculate_instances(gt["nuclei_type_map"], gt["instance_map"])

    batch_metrics, scores = calculate_step_metric(predictions, gt, num_nuclei_classes)
    batch_metrics["tissue_types"] = sample['tissue_type']

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

    model_cell_vit = prepare_model('/data/pwojcik/SimMIM/output_dir_800_b/checkpoint-latest.pth',
                                   init_values=args.init_values,
                                   drop_path_rate=args.drop_path,
                                   num_nuclei_classes=num_nuclei_classes,
                                   num_tissue_classes=num_tissue_classes,
                                   embed_dim=args.embed_dim,
                                   extract_layers=args.extract_layers)

    device = torch.device(args.device)

    model_cell_vit.to(device)
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

    binary_dice_scores = []  # binary dice scores per image
    binary_jaccard_scores = []  # binary jaccard scores per image
    pq_scores = []  # pq-scores per image
    dq_scores = []  # dq-scores per image
    sq_scores = []  # sq-scores per image
    cell_type_pq_scores = []  # pq-scores per cell type and image
    cell_type_dq_scores = []  # dq-scores per cell type and image
    cell_type_sq_scores = []  # sq-scores per cell type and image
    tissue_pred = []  # tissue predictions for each image
    tissue_gt = []  # ground truth tissue image class
    tissue_types_inf = []  # string repr of ground truth tissue image class

    paired_all_global = []  # unique matched index pair
    unpaired_true_all_global = (
        []
    )  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all_global = (
        []
    )  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all_global = []  # each index is 1 independent data point
    pred_inst_type_all_global = []  # each index is 1 independent data point

    # for detections scores
    true_idx_offset = 0
    pred_idx_offset = 0

    for idx, sample in tqdm(enumerate(dataloader_inference), total=len(dataloader_inference)):

        x = sample['x']

        batch_metrics, _ = compute_metrics(model_cell_vit, sample, num_nuclei_classes)

        # dice scores
        binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
        )
        binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
        )

        # pq scores
        pq_scores = pq_scores + batch_metrics["pq_scores"]
        dq_scores = dq_scores + batch_metrics["dq_scores"]
        sq_scores = sq_scores + batch_metrics["sq_scores"]
        tissue_types_inf = tissue_types_inf + batch_metrics["tissue_types"]
        cell_type_pq_scores = (
                cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
        )
        cell_type_dq_scores = (
                cell_type_dq_scores + batch_metrics["cell_type_dq_scores"]
        )
        cell_type_sq_scores = (
                cell_type_sq_scores + batch_metrics["cell_type_sq_scores"]
        )
        tissue_pred.append(batch_metrics["tissue_pred"])
        tissue_gt.append(batch_metrics["tissue_gt"])

        # detection scores
        true_idx_offset = (
            true_idx_offset + true_inst_type_all_global[-1].shape[0]
            if idx != 0
            else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all_global[-1].shape[0]
            if idx != 0
            else 0
        )
        true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
        pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
        # increment the pairing index statistic
        batch_metrics["paired_all"][:, 0] += true_idx_offset
        batch_metrics["paired_all"][:, 1] += pred_idx_offset
        paired_all_global.append(batch_metrics["paired_all"])

        batch_metrics["unpaired_true_all"] += true_idx_offset
        batch_metrics["unpaired_pred_all"] += pred_idx_offset
        unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
        unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

    # assemble batches to datasets (global)
    tissue_types_inf = [t.lower() for t in tissue_types_inf]

    paired_all = np.concatenate(paired_all_global, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    binary_dice_scores = np.array(binary_dice_scores)
    binary_jaccard_scores = np.array(binary_jaccard_scores)
    pq_scores = np.array(pq_scores)
    dq_scores = np.array(dq_scores)
    sq_scores = np.array(sq_scores)

    tissue_detection_accuracy = accuracy_score(
        y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
    )
    f1_d, prec_d, rec_d = cell_detection_scores(
        paired_true=paired_true_type,
        paired_pred=paired_pred_type,
        unpaired_true=unpaired_true_type,
        unpaired_pred=unpaired_pred_type,
    )
    dataset_metrics = {
        "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
        "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
        "Tissue-Multiclass-Accuracy": tissue_detection_accuracy,
        "bPQ": float(np.nanmean(pq_scores)),
        "bDQ": float(np.nanmean(dq_scores)),
        "bSQ": float(np.nanmean(sq_scores)),
        "mPQ": float(np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores])),
        "mDQ": float(np.nanmean([np.nanmean(dq) for dq in cell_type_dq_scores])),
        "mSQ": float(np.nanmean([np.nanmean(sq) for sq in cell_type_sq_scores])),
        "f1_detection": float(f1_d),
        "precision_detection": float(prec_d),
        "recall_detection": float(rec_d),
    }